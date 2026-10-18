"""Qwen LoRA policy used by both main and tiny debug runs."""

from __future__ import annotations

import json
import warnings
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class QwenGeneration:
    text: str
    input_tokens: int
    output_tokens: int
    generation_attempts: int = 1
    retry_count: int = 0
    length_retry_count: int = 0
    fallback_count: int = 0


@dataclass(frozen=True)
class QueryGeneration:
    queries: tuple[str, ...]
    input_tokens: int
    output_tokens: int
    generation_attempts: int = 1
    retry_count: int = 0
    length_retry_count: int = 0
    fallback_count: int = 0


class PlannerPolicy:
    def __init__(
        self,
        model_path: Path,
        device: str,
        *,
        learning_rate: float = 1e-5,
        seed: int = 0,
        learner: bool = True,
    ) -> None:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = torch.device(device)
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
            torch.cuda.manual_seed(seed)
        torch.manual_seed(seed)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        base = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to(self.device)
        base.config.use_cache = False
        self.model = get_peft_model(
            base,
            LoraConfig(
                r=4,
                lora_alpha=8,
                lora_dropout=0.0,
                target_modules=["q_proj", "v_proj"],
                task_type="CAUSAL_LM",
            ),
        )
        self.optimizer = (
            torch.optim.AdamW(self.trainable_parameters(), lr=learning_rate)
            if learner
            else None
        )
        self.actor_scoring_tokens = 0

    def _action_prefix(self, prompt: str) -> str:
        return self.tokenizer.apply_chat_template(
            [
                {
                    "role": "system",
                    "content": (
                        "You select the next action in a RAG workflow. Reply with "
                        'one JSON object only, using exactly {"action":"<action>"}.'
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def action_logprobs(self, prompt: str, labels: list[str], *, grad: bool) -> Any:
        return self.batch_action_logprobs([prompt], [labels], grad=grad)[0]

    def batch_action_logprobs(
        self,
        prompts: Sequence[str],
        label_groups: Sequence[Sequence[str]],
        *,
        grad: bool,
    ) -> list[Any]:
        """Score several planner states in one padded model forward."""

        if len(prompts) != len(label_groups) or not prompts:
            raise ValueError("one non-empty label group is required per prompt")
        torch = self.torch
        prefixes = [self._action_prefix(prompt) for prompt in prompts]
        prefix_lengths = [
            len(self.tokenizer(prefix, add_special_tokens=False)["input_ids"])
            for prefix in prefixes
        ]
        sequences: list[str] = []
        owners: list[int] = []
        for owner, (prefix, labels) in enumerate(
            zip(prefixes, label_groups, strict=True)
        ):
            if not labels:
                raise ValueError("planner state has no action labels")
            for label in labels:
                action_json = json.dumps(
                    {"action": str(label)},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                sequences.append(prefix + action_json)
                owners.append(owner)
        encoded = self.tokenizer(
            sequences,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        context = nullcontext() if grad else torch.no_grad()
        with context:
            logits = self.model(**encoded, use_cache=False).logits
            grouped: list[list[Any]] = [[] for _ in prompts]
            lengths = encoded.attention_mask.sum(dim=1).tolist()
            self.actor_scoring_tokens += sum(int(length) for length in lengths)
            for row, (owner, length) in enumerate(zip(owners, lengths, strict=True)):
                suffix_length = int(length) - prefix_lengths[owner]
                if suffix_length <= 0:
                    raise RuntimeError("JSON action verbalizer has no tokens")
                suffix_logits = logits[
                    row,
                    prefix_lengths[owner] - 1 : int(length) - 1,
                ].float()
                suffix_tokens = encoded.input_ids[
                    row,
                    prefix_lengths[owner] : int(length),
                ]
                grouped[owner].append(
                    -torch.nn.functional.cross_entropy(
                        suffix_logits,
                        suffix_tokens,
                        reduction="mean",
                    )
                )
            return [torch.stack(values) for values in grouped]

    def distribution(
        self,
        prompt: str,
        labels: list[str],
    ) -> tuple[list[float], list[float]]:
        scores = self.action_logprobs(prompt, labels, grad=False)
        logprobabilities = self.torch.log_softmax(scores, dim=0)
        return (
            logprobabilities.exp().cpu().tolist(),
            logprobabilities.cpu().tolist(),
        )

    def probabilities(self, prompt: str, labels: list[str]) -> list[float]:
        probabilities, _ = self.distribution(prompt, labels)
        return probabilities

    @staticmethod
    def _first_json_object(text: str) -> dict[str, Any] | None:
        decoder = json.JSONDecoder()
        for index, character in enumerate(text):
            if character != "{":
                continue
            try:
                value, _ = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
        return None

    def _generate_chat_json(
        self,
        messages: list[dict[str, str]],
        *,
        max_new_tokens: int,
    ) -> QwenGeneration:
        torch = self.torch
        encoded = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
        )
        input_ids = (
            encoded.input_ids if hasattr(encoded, "input_ids") else encoded
        ).to(self.device)
        attention_mask = torch.ones_like(input_ids)
        with self.model.disable_adapter(), torch.inference_mode():
            generated = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        generated_tokens = generated[0, input_ids.shape[1] :]
        return QwenGeneration(
            text=self.tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True,
            ).strip(),
            input_tokens=int(input_ids.shape[1]),
            output_tokens=int(generated_tokens.shape[0]),
        )

    @staticmethod
    def _grounded_answer_messages(
        question: str,
        evidence: Sequence[Any],
        *,
        retry_index: int = 0,
    ) -> tuple[list[dict[str, str]], set[str]]:
        evidence_text = "\n\n".join(
            "\n".join(
                (
                    f"citation_handle=E{index}",
                    f"title={item.title}",
                    f"content={item.text}",
                )
            )
            for index, item in enumerate(evidence, start=1)
        )
        allowed_citations = {f"E{index}" for index in range(1, len(evidence) + 1)}
        user_content = (
            "Answer only from the evidence. Return the shortest answer span "
            "that satisfies the question. Use exactly the fields answer "
            "(a non-empty string) and citations (a non-empty array containing "
            "only handles such as E1). If the evidence cannot determine the "
            "answer, use answer UNKNOWN and cite the most relevant evidence."
            f"\n\nQuestion:\n{question}\n\nEvidence:\n{evidence_text}"
        )
        if retry_index:
            user_content += (
                f"\n\nRetry {retry_index}: the previous reply was invalid or "
                "incomplete. Return only the required JSON object."
            )
        return (
            [
                {
                    "role": "system",
                    "content": (
                        "You are a grounded question-answering generator. Output "
                        "one JSON object and no markdown. Citation values must be "
                        "exact case-sensitive evidence handles."
                    ),
                },
                {"role": "user", "content": user_content},
            ],
            allowed_citations,
        )

    def estimate_grounded_answer_tokens(
        self,
        question: str,
        evidence: Sequence[Any],
        *,
        expected_output_tokens: int = 64,
    ) -> int:
        """Estimate a valid first-attempt generation in raw Qwen-token units."""

        if not evidence:
            raise ValueError("grounded generation token estimate requires evidence")
        messages, _ = self._grounded_answer_messages(question, evidence)
        encoded = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        input_ids = encoded.input_ids if hasattr(encoded, "input_ids") else encoded
        input_tokens = (
            len(input_ids[0])
            if input_ids and isinstance(input_ids[0], list)
            else len(input_ids)
        )
        return input_tokens + expected_output_tokens

    def generate_grounded_answer(
        self,
        question: str,
        evidence: Sequence[Any],
        *,
        max_new_tokens: int = 768,
    ) -> QwenGeneration:
        """Generate one grounded answer with the frozen base Qwen model."""

        if not evidence:
            raise ValueError("grounded generation requires evidence")
        base_messages, allowed_citations = self._grounded_answer_messages(
            question, evidence
        )
        last_output = ""
        length_retry_count = 0
        total_input_tokens = 0
        total_output_tokens = 0
        attempt_max_new_tokens = max_new_tokens
        max_attempts = 3
        for attempt in range(max_attempts):
            messages = (
                base_messages
                if attempt == 0
                else self._grounded_answer_messages(
                    question, evidence, retry_index=attempt
                )[0]
            )
            current_max_new_tokens = attempt_max_new_tokens
            generated = self._generate_chat_json(
                messages,
                max_new_tokens=current_max_new_tokens,
            )
            total_input_tokens += generated.input_tokens
            total_output_tokens += generated.output_tokens
            last_output = generated.text
            value = self._first_json_object(last_output)
            if value is None:
                if (
                    generated.output_tokens >= current_max_new_tokens
                    and attempt + 1 < max_attempts
                ):
                    length_retry_count += 1
            else:
                answer = value.get("answer")
                citations = value.get("citations")
                if (
                    set(value) == {"answer", "citations"}
                    and isinstance(answer, str)
                    and answer.strip()
                    and isinstance(citations, list)
                    and bool(citations)
                    and all(
                        isinstance(citation, str) and citation in allowed_citations
                        for citation in citations
                    )
                ):
                    return QwenGeneration(
                        text=answer.strip(),
                        input_tokens=total_input_tokens,
                        output_tokens=total_output_tokens,
                        generation_attempts=attempt + 1,
                        retry_count=attempt,
                        length_retry_count=length_retry_count,
                    )
            if attempt + 1 < max_attempts:
                attempt_max_new_tokens = max(
                    current_max_new_tokens + 128,
                    (current_max_new_tokens * 3 + 1) // 2,
                )
        warnings.warn(
            "Qwen did not return the required grounded-answer JSON template; "
            f"using UNKNOWN after {max_attempts} attempts "
            f"({max_attempts - 1} retries, {length_retry_count} "
            f"length-triggered): {last_output[:240]}",
            RuntimeWarning,
            stacklevel=2,
        )
        return QwenGeneration(
            text="UNKNOWN",
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            generation_attempts=max_attempts,
            retry_count=max_attempts - 1,
            length_retry_count=length_retry_count,
            fallback_count=1,
        )

    @staticmethod
    def _workflow_query_messages(
        strategy: str,
        question: str,
        *,
        retry_index: int = 0,
    ) -> list[dict[str, str]]:
        if strategy == "query.query2doc":
            instruction = (
                "Write a short hypothetical passage that would answer the question. "
                "Return exactly one JSON object with the single string field "
                '"pseudo_document".'
            )
        elif strategy == "query.decomposition":
            instruction = (
                "Decompose the multi-hop question into exactly two focused search "
                "queries. Return exactly one JSON object with the single field "
                '"queries", an array of exactly two non-empty strings.'
            )
        else:
            raise ValueError(f"unknown workflow query strategy: {strategy}")
        if retry_index:
            instruction += (
                f" Retry {retry_index}: the previous reply was invalid or incomplete; "
                "return only the required JSON object."
            )
        return [
            {
                "role": "system",
                "content": (
                    "You create retrieval queries for a RAG workflow. Output one JSON "
                    "object only, with no markdown or extra fields."
                ),
            },
            {"role": "user", "content": f"{instruction}\n\nQuestion:\n{question}"},
        ]

    def estimate_workflow_query_tokens(
        self,
        strategy: str,
        question: str,
        *,
        expected_output_tokens: int = 96,
    ) -> int:
        if strategy == "query.identity":
            return 0
        messages = self._workflow_query_messages(strategy, question)
        encoded = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        input_ids = encoded.input_ids if hasattr(encoded, "input_ids") else encoded
        input_tokens = (
            len(input_ids[0])
            if input_ids and isinstance(input_ids[0], list)
            else len(input_ids)
        )
        return input_tokens + expected_output_tokens

    def generate_workflow_query(
        self,
        strategy: str,
        question: str,
        *,
        max_new_tokens: int = 384,
    ) -> QueryGeneration:
        """Generate Query2doc or two decomposition queries using a JSON template."""

        if strategy == "query.identity":
            return QueryGeneration(
                queries=(question,),
                input_tokens=0,
                output_tokens=0,
                generation_attempts=0,
            )
        total_input_tokens = 0
        total_output_tokens = 0
        length_retry_count = 0
        attempt_max_new_tokens = max_new_tokens
        last_output = ""
        max_attempts = 3
        for attempt in range(max_attempts):
            generated = self._generate_chat_json(
                self._workflow_query_messages(
                    strategy, question, retry_index=attempt
                ),
                max_new_tokens=attempt_max_new_tokens,
            )
            total_input_tokens += generated.input_tokens
            total_output_tokens += generated.output_tokens
            last_output = generated.text
            value = self._first_json_object(last_output)
            queries: tuple[str, ...] = ()
            if value is not None:
                if strategy == "query.query2doc":
                    document = value.get("pseudo_document")
                    if set(value) == {"pseudo_document"} and isinstance(document, str):
                        queries = (document.strip(),) if document.strip() else ()
                else:
                    values = value.get("queries")
                    if (
                        set(value) == {"queries"}
                        and isinstance(values, list)
                        and len(values) == 2
                        and all(isinstance(query, str) and query.strip() for query in values)
                    ):
                        queries = tuple(query.strip() for query in values)
            if queries:
                return QueryGeneration(
                    queries=queries,
                    input_tokens=total_input_tokens,
                    output_tokens=total_output_tokens,
                    generation_attempts=attempt + 1,
                    retry_count=attempt,
                    length_retry_count=length_retry_count,
                )
            if generated.output_tokens >= attempt_max_new_tokens:
                length_retry_count += 1
            attempt_max_new_tokens = max(
                attempt_max_new_tokens + 128,
                (attempt_max_new_tokens * 3 + 1) // 2,
            )
        warnings.warn(
            "Qwen did not return the required retrieval-query JSON template; "
            f"falling back to the original question: {last_output[:240]}",
            RuntimeWarning,
            stacklevel=2,
        )
        return QueryGeneration(
            queries=(question,),
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            generation_attempts=max_attempts,
            retry_count=max_attempts - 1,
            length_retry_count=length_retry_count,
            fallback_count=1,
        )

    def adapter_state(self) -> dict[str, Any]:
        from peft import get_peft_model_state_dict

        return {
            key: value.detach().cpu()
            for key, value in get_peft_model_state_dict(self.model).items()
        }

    def load_adapter_state(self, state: dict[str, Any]) -> None:
        from peft import set_peft_model_state_dict

        set_peft_model_state_dict(self.model, state)

    def optimizer_state(self) -> dict[str, Any]:
        if self.optimizer is None:
            raise RuntimeError("collection-only policy has no optimizer")
        return self.optimizer.state_dict()

    def load_optimizer_state(self, state: dict[str, Any]) -> None:
        if self.optimizer is None:
            raise RuntimeError("collection-only policy has no optimizer")
        self.optimizer.load_state_dict(state)
        for values in self.optimizer.state.values():
            for key, value in values.items():
                if self.torch.is_tensor(value):
                    values[key] = value.to(self.device)

    def trainable_parameters(self) -> list[Any]:
        return [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad
        ]
