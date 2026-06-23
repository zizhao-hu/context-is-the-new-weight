"""windowed_attention.png -- 2x2 paper figure. causal / sliding+history / sliding+trainable are attention masks
(rows = a prediction step, cols = attended positions). UNIFORM IS NOT AN ATTENTION MAP: it is fixed-length
predict-last, drawn as four separate windows, each W real context tokens (green) predicting only its last/next
token (crimson, col c6). All cells are equal squares; the first (left) column is left-aligned."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
CONTENT,HISTC,PROMPT,PREDC=(.12,.47,.71),(.12,.47,.71),(.93,.50,.11),(.74,.13,.13)
GREY=(.74,.76,.79)
CTX,PRD="#555","#b21c1c"
fig,ax=plt.subplots(figsize=(8.6,6.3)); ax.set_aspect("equal"); ax.axis("off")
def Cl(x,y,fc): ax.add_patch(Rectangle((x,y),1,1,facecolor=fc,edgecolor="#555",lw=1.3,zorder=2))
def Em(x,y): ax.add_patch(Rectangle((x,y),1,1,facecolor="white",edgecolor="#cfcfcf",lw=0.7,zorder=1))
def TT(x,y,t): ax.text(x,y,t,fontsize=13.5,fontweight="bold",ha="left",va="bottom")
def XL(x,y,t,c): ax.text(x+.5,y,t,fontsize=10,fontweight="bold",ha="center",va="top",color=c)
def YL(x,y,t,c): ax.text(x,y+.5,t,fontsize=10,fontweight="bold",ha="right",va="center",color=c)
PRED=["→t2","→t3","→t4","→t5","→t6"]; P,Wl=2,3
def attn(ox,oy,title,fill,xlabels):
    TT(ox,oy-.35,title)
    NC=P+5
    for r in range(5):
        hi=P+r; lo=hi-(Wl-1)        # window of Wl ending at the most-recent content token; older tokens scroll out
        for c in range(NC):
            if lo<=c<=hi: Cl(ox+c,oy+r, CONTENT if c>=P else fill)
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
attn(9.5,0,"sliding + history",HISTC,["t-1","t0","t1","t2","t3","t4","t5"])
# UNIFORM (NOT attention) -- two example sequences shifted along the stream + vertical dots = stacking dataset
ox,oy=0,6.2; TT(ox,oy-.35,"uniform (constant $W$)")
def useq(sx,ry,toks):           # 5-length predict-last sequence; tokens by ABSOLUTE position; labels INSIDE cells
    for c in range(4):
        Cl(ox+sx+c,ry,GREY); ax.text(ox+sx+c+.5,ry+.5,toks[c],fontsize=10.5,fontweight="bold",ha="center",va="center",color="#222")
    Cl(ox+sx+4,ry,PREDC); ax.text(ox+sx+4+.5,ry+.5,toks[4],fontsize=10.5,fontweight="bold",ha="center",va="center",color="white")
useq(0,6.2,["t1","t2","t3","t4","t5"])      # top seq aligns with the right panel's top row (->t2)
ax.text(ox+2.5,8.7,r"$\vdots$",fontsize=22,fontweight="bold",ha="center",va="center")
useq(0,10.2,["t3","t4","t5","t6","t7"])     # bottom seq aligns with bottom row; numbers shifted
# SLIDING+TRAINABLE -- right col, bottom
attn(9.5,6.2,"sliding + trainable",PROMPT,["p1","p2","t1","t2","t3","t4","t5"])
# no legend: label context / predicted / trainable prompt with brackets under the example cells
def brace(x0,x1,label,ytop,col):
    yb=ytop+0.26
    ax.plot([x0+.06,x0+.06,x1-.06,x1-.06],[ytop+.05,yb,yb,ytop+.05],color=col,lw=1.8,zorder=5)
    ax.text((x0+x1)/2,yb+.12,label,fontsize=10.5,fontweight="bold",ha="center",va="top",color=col)
brace(0,4,"context",11.2,"#555")                 # grey cells of bottom uniform seq
brace(4,5,"predicted",11.2,"#b21c1c")            # crimson cell
brace(9.5,11.5,"trainable prompt",11.45,"#cf7f1a")  # p1,p2 columns of sliding+trainable
ax.set_xlim(-1.45,16.7); ax.set_ylim(12.6,-0.95)
fig.subplots_adjust(left=0.004,right=0.996,top=0.996,bottom=0.004)
out="/Users/zizhaohu/Desktop/projects/context-is-the-new-weight/.claude/worktrees/exp/paper/attention-sink/figures/windowed_attention.png"
plt.savefig(out,dpi=175,bbox_inches="tight",pad_inches=0.03); print("wrote",out)
