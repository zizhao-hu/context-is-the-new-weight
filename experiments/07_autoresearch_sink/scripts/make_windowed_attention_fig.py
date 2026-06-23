"""windowed_attention.png -- 2x2 paper figure. causal / sliding+history / sliding+trainable are attention masks
(rows = a prediction step, cols = attended positions). UNIFORM IS NOT AN ATTENTION MAP: it is fixed-length
predict-last, drawn as four separate windows, each W real context tokens (green) predicting only its last/next
token (crimson, col c6). All cells are equal squares; the first (left) column is left-aligned."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
CONTENT,HISTC,PROMPT,PREDC=(.12,.47,.71),(.12,.47,.71),(.93,.50,.11),(.74,.13,.13)
CTX,PRD="#555","#b21c1c"
fig,ax=plt.subplots(figsize=(8.8,6.0)); ax.set_aspect("equal"); ax.axis("off")
def Cl(x,y,fc): ax.add_patch(Rectangle((x,y),1,1,facecolor=fc,edgecolor="#555",lw=1.3,zorder=2))
def Em(x,y): ax.add_patch(Rectangle((x,y),1,1,facecolor="white",edgecolor="#cfcfcf",lw=0.7,zorder=1))
def TT(x,y,t): ax.text(x,y,t,fontsize=13.5,fontweight="bold",ha="left",va="bottom")
def XL(x,y,t,c): ax.text(x+.5,y,t,fontsize=10,fontweight="bold",ha="center",va="top",color=c)
def YL(x,y,t,c): ax.text(x,y+.5,t,fontsize=10,fontweight="bold",ha="right",va="center",color=c)
PRED=["→t2","→t3","→t4","→t5","→t6"]; P,Wl=5,6
def attn(ox,oy,title,fill,xlabels):
    TT(ox,oy-.35,title)
    for r in range(5):
        i=r+1
        for c in range(10):
            if P<=c<P+i: Cl(ox+c,oy+r,CONTENT)
            elif c<P and c>=P-(Wl-i): Cl(ox+c,oy+r,fill)
            else: Em(ox+c,oy+r)
        YL(ox-.25,oy+r,PRED[r],PRD)
    for c,t in enumerate(xlabels): XL(ox+c,oy+5+.12,t,CTX)
# CAUSAL (content side only, 5x5)  -- left col, top
ox,oy=0,0; TT(ox,oy-.35,"causal (grows)")
for r in range(5):
    for c in range(5): (Cl(ox+c,oy+r,CONTENT) if c<=r else Em(ox+c,oy+r))
    YL(ox-.25,oy+r,PRED[r],PRD)
for c,t in enumerate(["t1","t2","t3","t4","t5"]): XL(ox+c,oy+5+.12,t,CTX)
# SLIDING+HISTORY  -- right col, top
attn(7.8,0,"sliding + history",HISTC,["t-4","t-3","t-2","t-1","t0","t1","t2","t3","t4","t5"])
# UNIFORM (NOT attention) -- four separate predict-last windows, left col, bottom, left-aligned w/ causal
ox,oy=0,7.2; g=1.0/3; TT(ox,oy-.35,"uniform (constant $W$)")
for w in range(4):
    ry=oy+w*(1+g)
    for c in range(5): Cl(ox+c,ry,CONTENT)
    Cl(ox+5,ry,PREDC)
    YL(ox-.35,ry,f"s{w+1}",CTX)
ybot=oy+3*(1+g)+1
for c,t in enumerate(["c1","c2","c3","c4","c5"]): XL(ox+c,ybot+.12,t,CTX)
XL(ox+5,ybot+.12,"c6",PRD)
# SLIDING+TRAINABLE -- right col, bottom
attn(7.8,7.2,"sliding + trainable",PROMPT,["p1","p2","p3","p4","p5","t1","t2","t3","t4","t5"])
ax.set_xlim(-1.9,18.3); ax.set_ylim(13.6,-1.5)
fig.legend(handles=[Patch(color=CONTENT,label="context"),Patch(color=PROMPT,label="trainable prompt"),
                    Patch(color=PREDC,label="predicted")],
           loc="upper center",ncol=3,fontsize=10.5,bbox_to_anchor=(0.5,1.0),frameon=False,handlelength=1.0,columnspacing=1.4,handletextpad=0.5)
out="/Users/zizhaohu/Desktop/projects/context-is-the-new-weight/.claude/worktrees/exp/paper/attention-sink/figures/windowed_attention.png"
plt.savefig(out,dpi=170,bbox_inches="tight"); print("wrote",out)
