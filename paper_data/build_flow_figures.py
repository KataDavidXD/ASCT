"""Deterministic vector reconstruction of the reviewed scientific-schematics concepts.

Illustrative numbers demonstrate credit construction, not experimental outcomes.
"""
from pathlib import Path
import hashlib,json,subprocess
import matplotlib
from plot_style import paper_font
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch,Circle,Rectangle

HERE=Path(__file__).resolve().parent
OUT=HERE/'figures'
plt.rcParams.update({'font.family':'serif','font.serif':[paper_font()],
    'font.size':10,'mathtext.fontset':'stix','pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
INK='#202831';BLUE='#16577C';GRAY='#586775';BROWN='#90611E';PALE='#EDF4F8'

def canvas(height):
    fig=plt.figure(figsize=(5.5,height),facecolor='white')
    ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,5.5),ylim=(0,height));ax.axis('off')
    return fig,ax

def text(ax,x,y,s,**kw):
    return ax.text(x,y,s,ha=kw.pop('ha','center'),va='center',fontsize=10,color=kw.pop('color',INK),zorder=6,**kw)

def arrow(ax,a,b,*,dash=False,color=INK,width=.85,curve=0):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=6,linewidth=width,color=color,
        linestyle=(0,(3,2)) if dash else '-',shrinkA=2,shrinkB=3,connectionstyle=f'arc3,rad={curve}',zorder=3))

def line(ax,pts,*,dash=False,color=INK,width=.8):
    ax.plot([p[0] for p in pts],[p[1] for p in pts],color=color,lw=width,ls=(0,(3,2)) if dash else '-',zorder=2)

def node(ax,p,label=None,*,fill='white',color=INK,r=.035):
    ax.add_patch(Circle(p,r,facecolor=fill,edgecolor=color,lw=.8,zorder=5))
    if label:text(ax,p[0],p[1],label,color=color)

def panel(ax,x,y,w,h,fill='white',edge='#C9D3DA'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.01,rounding_size=0.04',fc=fill,ec=edge,lw=.7,zorder=0))

def save(fig,ax,name):
    fig.canvas.draw();ren=fig.canvas.get_renderer()
    for artist in ax.texts:
        bb=artist.get_window_extent(ren).transformed(fig.transFigure.inverted())
        assert bb.x0>=-.001 and bb.x1<=1.001 and bb.y0>=0 and bb.y1<=1,(name,artist.get_text(),'canvas overflow',bb.bounds)
    OUT.mkdir(exist_ok=True)
    for ext in ('pdf','svg','png'):
        fig.savefig(OUT/f'{name}.{ext}',dpi=300,facecolor='white',metadata={'Creator':'ASCT scientific mechanism figures'} if ext=='pdf' else None)
    fonts=subprocess.run(['pdffonts',str(OUT/f'{name}.pdf')],capture_output=True,text=True,check=True).stdout
    assert 'Type 3' not in fonts and all(r.split()[-5]=='yes' for r in fonts.splitlines()[2:] if r.strip())
    report=dict(width_inches=5.5,height_inches=float(fig.get_figheight()),minimum_font_pt=10,
        all_fonts_embedded=True,type3_fonts=False,text_canvas_bounds_checked=True,
        pdf_sha256=hashlib.sha256((OUT/f'{name}.pdf').read_bytes()).hexdigest())
    plt.close(fig)
    return report

def training():
    fig,ax=canvas(2.95)
    original_transform=ax.transData
    from matplotlib.transforms import Affine2D
    ax.transData=Affine2D().translate(0,-1.20)+original_transform
    text(ax,.08,4.02,'(a) ASCT: compare actions at the same state',ha='left',fontweight='bold',color=BLUE)
    # The actual sampled action, not a search-selected action, advances the actor.
    points=[(.22,3.66),(.92,3.66),(1.94,3.66)]
    for a,b in zip(points,points[1:]):arrow(ax,a,b,width=1.1)
    node(ax,points[0]);node(ax,points[1],r=.075,fill=PALE,color=BLUE);node(ax,points[2])
    text(ax,.92,3.66,r'$s$',color=BLUE)
    text(ax,1.44,3.84,r'$a_2$ sampled')
    arrow(ax,points[2],(4.22,3.66),width=1.1)
    text(ax,3.08,3.83,'Main actor records')
    panel(ax,4.22,3.53,.53,.26);text(ax,4.485,3.66,'PPO')
    arrow(ax,(4.77,3.66),(5.02,3.66));text(ax,5.23,3.66,'Actor\nonly',linespacing=1)
    # Same-state root-action coverage plus multiple terminal continuations.
    panel(ax,.08,2.22,2.25,1.14,fill=PALE)
    text(ax,1.20,3.21,'Restore / reuse prefix',color=BLUE)
    root=(.28,2.78);node(ax,root,r=.06,fill=PALE,color=BLUE)
    line(ax,[(.92,3.52),(.28,3.52)],dash=True,color=BLUE)
    arrow(ax,(.28,3.52),root,dash=True,color=BLUE)
    text(ax,.28,2.78,r'$s$',color=BLUE)
    for i,y in enumerate([3.01,2.75,2.49]):
        child=(.89,y);mid=(1.40,y)
        arrow(ax,root,child,dash=True,color=BLUE);node(ax,child,color=BLUE)
        text(ax,.73,y+.09,f'$a_{i+1}$',color=BLUE)
        arrow(ax,child,mid,dash=True,color=BLUE);node(ax,mid,color=BLUE)
        for dy in [-.065,.065]:
            leaf=(1.91,y+dy);arrow(ax,mid,leaf,dash=True,color=BLUE);node(ax,leaf,fill=BROWN,color=BROWN,r=.027)
    text(ax,1.18,2.33,'Multi-step terminal outcomes',color=BLUE)
    arrow(ax,(2.05,2.79),(2.46,2.79),dash=True,color=BLUE)
    # Q table and actor probabilities are visibly separate quantities.
    text(ax,3.02,3.23,'Action table',fontweight='bold',color=BLUE)
    for x,s in [(2.66,'Action'),(3.13,r'$\widehat Q$'),(3.58,r'$p$')]:text(ax,x,3.03,s)
    line(ax,[(2.43,2.92),(3.76,2.92)],color=GRAY)
    ax.add_patch(Rectangle((2.43,2.46),1.33,.27,fc='#D8E9F3',ec='none',zorder=0))
    for i,(q,p,y) in enumerate([(.40,.20,2.80),(.60,.30,2.59),(.90,.50,2.37)]):
        for x,s in [(2.66,f'$a_{i+1}$'),(3.13,f'{q:.2f}'),(3.58,f'{p:.2f}')]:text(ax,x,y,s,fontweight='bold' if i==1 else 'normal')
    text(ax,3.03,2.15,'Illustrative values',fontstyle='italic')
    arrow(ax,(3.78,2.80),(3.99,2.80),dash=True,color=BLUE)
    # Credit is the gap relative to the actor mixture, not the sign of utility.
    text(ax,4.65,3.23,'Actor-relative credit',fontweight='bold',color=BLUE)
    for y,val,label,color in [(2.92,.60,r'$\widehat Q(a_2)=0.60$',BLUE),(2.57,.71,r'$\sum_a p_a\widehat Q(a)=0.71$',BROWN)]:
        text(ax,4.65,y+.13,label)
        ax.add_patch(Rectangle((4.01,y-.045),1.65*val,.075,fc=color,ec=color,lw=.4))
    line(ax,[(4.01+1.65*.71,2.46),(4.01+1.65*.71,3.01)],dash=True,color=BROWN)
    text(ax,4.66,2.24,r'$\widehat A(a_2)=-0.11$',color=BLUE,fontweight='bold')
    arrow(ax,(4.55,3.38),(1.49,3.66),dash=True,color=BLUE,curve=-.10)
    text(ax,3.10,3.43,'Credit only',color=BLUE)
    line(ax,[(.08,1.96),(5.42,1.96)],color='#B3BFC7')
    ax.transData=original_transform
    text(ax,.08,.62,'(b) VinePPO: MC state values',ha='left',fontweight='bold',color=BROWN)
    for x in [.26,.98]:
        node(ax,(x,.35),fill='#F6EEE0',color=BROWN,r=.06)
        text(ax,x,.35,r'$s$' if x<.5 else r'$s^{\prime}$',color=BROWN)
        for dx in [-.13,0,.13]:
            leaf=(x+dx,.13);arrow(ax,(x,.31),leaf,dash=True,color=BROWN);node(ax,leaf,fill=BROWN,color=BROWN,r=.018)
    arrow(ax,(.31,.35),(.91,.35))
    arrow(ax,(1.06,.35),(1.39,.35))
    text(ax,1.78,.36,r'$r+\widehat V(s^{\prime})-\widehat V(s)$')
    text(ax,2.13,.12,'Main actor → PPO')
    line(ax,[(2.78,.06),(2.78,.72)],color='#B3BFC7')
    text(ax,2.91,.62,r'(c) AT$^2$PO: branch credit',ha='left',fontweight='bold',color=GRAY)
    root=(3.01,.28);node(ax,root,r=.025)
    for y in [.15,.41]:
        child=(3.33,y);arrow(ax,root,child);node(ax,child,r=.025)
        for dy in [-.05,.05]:
            leaf=(3.64,y+dy);arrow(ax,child,leaf);node(ax,leaf,fill=BROWN,color=BROWN,r=.022)
    arrow(ax,(3.77,.28),(4.03,.28))
    text(ax,4.76,.30,'Branched records\n→ policy update',linespacing=1.15)
    return save(fig,ax,'training_interface_flow')

def evaluators():
    fig,ax=canvas(3.35)
    text(ax,2.75,3.21,'Same restored state · Same trial budget · Shared prefix reuse',fontweight='bold')
    # One next-edge decision differs at the same internal node. Width is a
    # categorical selected-edge marker, never a visit-count/effect-size encoding.
    for i,(name,color) in enumerate([('Uniform',GRAY),('UCT',GRAY),('AgentUCT',GRAY)]):
        x=.11+i*1.82
        if i:line(ax,[(x-.05,.65),(x-.05,2.96)],color='#C9D3DA')
        text(ax,x+.82,2.94,name,color=color,fontweight='bold')
        root=(x+.82,2.60);node(ax,root,fill='#DDE2E6',r=.06)
        for j,cx in enumerate([x+.27,x+.82,x+1.37]):
            child=(cx,2.12);arrow(ax,root,child,color=color,width=1.0);node(ax,child,fill='#DDE2E6' if j==1 else 'white',color=color,r=.045)
            for dx in [-.15,.15]:
                leaf=(cx+dx,1.68)
                selected=j==2 and ((i==1 and dx>0) or (i==2 and dx<0))
                arrow(ax,child,leaf,color=BLUE if selected else color,width=2.0 if selected else .8)
                node(ax,leaf,fill=color,color=color,r=.026)
            if i>0 and j==2:
                # Same inspected decision state in UCT and AgentUCT.
                ax.add_patch(Circle(child,.065,fc='none',ec=BLUE,lw=.85,zorder=6))
        if i>0:
            text(ax,x+.82,1.45,'One local choice',color=BLUE)
        if i==0:
            text(ax,x+.82,1.20,'Balance visits')
            text(ax,x+.82,.86,'Uniform expansion\nand rollout',linespacing=1.1)
        elif i==1:
            text(ax,x+.82,1.20,'Utility + exploration')
            text(ax,x+.82,.86,'Uniform expansion\nand rollout',linespacing=1.1)
        else:
            text(ax,x+.82,1.20,'Utility + exploration\n− uncached cost',linespacing=1.05)
            text(ax,x+.82,.86,'Cost-weighted expansion\nand rollout',linespacing=1.1)
        arrow(ax,(x+.82,.63),(x+.82,.49),color=color)
    panel(ax,.10,.20,5.30,.28,fill=PALE,edge=BLUE)
    text(ax,2.75,.34,r'Same output: root-action $\widehat Q$ means from terminal utility $U$',color=BLUE)
    text(ax,2.75,.08,'Thick arrow: next choice (illustrative); shaded nodes: reusable prefixes')
    return save(fig,ax,'evaluator_flow')

def main():
    assert abs(sum(q*p for q,p in zip([.4,.6,.9],[.2,.3,.5]))-.71)<1e-12
    figures={'training_interface_flow':training(),'evaluator_flow':evaluators()}
    (HERE/'flow_audit.json').write_text(json.dumps(dict(status='passed',figures=figures,
        toy_example={'Q':[.4,.6,.9],'p':[.2,.3,.5],'baseline':.71,'sampled_action':2,'credit':-.11},
        semantic_scope='Workflow implementations; toy Q table is illustrative. UCT and AgentUCT differ only in the highlighted next edge at one common internal node; width is categorical, not visitation or effect size.',
        evaluator_drawing={'identical_tree_topology':True,'differing_decision_nodes':1,'edge_width_encodes':'one illustrative selected edge','cost_effect_size_encoded':False}),indent=2))
    print('Mechanism figures: vector PDF/SVG/PNG, 10 pt embedded type, verified illustrative arithmetic')

if __name__=='__main__':main()
