"""windowed_attention.png -- 2x2 paper figure, methods.html scheme style. Axis token labels are colored by role:
grey = context (attended), crimson = predicted (loss). causal/sliding/trainable predict EVERY token (each row a
prediction); uniform predicts ONLY its last (extra) column (fixed-length predict-last) and shows the full constant
window as a solid block. causal & uniform have no cold-start fill, so only the content side is drawn."""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
CONTENT, HISTC, PROMPT, PREDC = (.18,.53,.31),(.60,.81,.67),(.93,.50,.11),(.74,.13,.13)
CTX, PRD = "#555", "#b21c1c"
def gl(a,NC,NR):
    for x in range(NC+1): a.plot([x-.5,x-.5],[-.5,NR-.5],color="#777",lw=1.0,zorder=3)
    for y in range(NR+1): a.plot([-.5,NC-.5],[y-.5,y-.5],color="#777",lw=1.0,zorder=3)
def fin(a,NC,NR,title,xl,xc,yl,yc):
    a.set_xlim(-.5,NC-.5); a.set_ylim(NR-.5,-.5)
    a.set_xticks(range(NC)); 
    for t,c in zip(a.set_xticklabels(xl,fontsize=9.5,fontweight="bold"),xc): t.set_color(c)
    a.set_yticks(range(NR))
    for t,c in zip(a.set_yticklabels(yl,fontsize=9.5,fontweight="bold"),yc): t.set_color(c)
    a.tick_params(length=0,pad=2.5); a.set_title(title,fontsize=12.5,fontweight="bold",pad=5)
    for s in a.spines.values(): s.set_linewidth(2.0); s.set_color("#222")
PRED=["→t2","→t3","→t4","→t5","→t6"]
fig=plt.figure(figsize=(8.6,4.7))
gs=gridspec.GridSpec(2,2,width_ratios=[6,10],hspace=0.6,wspace=0.16)
# causal -- right part only
a=fig.add_subplot(gs[0,0]); img=np.zeros((5,5,4))
for r in range(5):
    for c in range(r+1): img[r,c]=(*CONTENT,1)
a.imshow(img,aspect="equal",interpolation="nearest",zorder=2); gl(a,5,5)
fin(a,5,5,"causal (grows)",["t1","t2","t3","t4","t5"],[CTX]*5,PRED,[PRD]*5); a.set_ylabel("predicting ↓",fontsize=9.5,fontweight="bold")
# sliding + history
a=fig.add_subplot(gs[0,1]); img=np.zeros((5,10,4)); P,Wl=5,6
for r in range(5):
    i=r+1
    for idx in range(i): img[r,P+idx]=(*CONTENT,1)
    for k in range(Wl-i): img[r,P-(Wl-i)+k]=(*HISTC,1)
a.imshow(img,aspect="equal",interpolation="nearest",zorder=2); gl(a,10,5)
fin(a,10,5,"sliding + history",["t-4","t-3","t-2","t-1","t0","t1","t2","t3","t4","t5"],[CTX]*10,PRED,[PRD]*5)
# uniform -- solid full block + predicted extra last column
a=fig.add_subplot(gs[1,0]); img=np.zeros((5,6,4))
for r in range(5):
    for c in range(5): img[r,c]=(*CONTENT,1)
img[4,5]=(*PREDC,1)
a.imshow(img,aspect="equal",interpolation="nearest",zorder=2); gl(a,6,5)
fin(a,6,5,"uniform (constant $W$)",["c1","c2","c3","c4","c5","c6"],[CTX]*5+[PRD],["c1","c2","c3","c4","c5"],[CTX]*5); a.set_ylabel("window pos.",fontsize=9.5,fontweight="bold")
# sliding + trainable
a=fig.add_subplot(gs[1,1]); img=np.zeros((5,10,4))
for r in range(5):
    i=r+1
    for idx in range(i): img[r,P+idx]=(*CONTENT,1)
    for k in range(Wl-i): img[r,P-(Wl-i)+k]=(*PROMPT,1)
a.imshow(img,aspect="equal",interpolation="nearest",zorder=2); gl(a,10,5)
fin(a,10,5,"sliding + trainable",["s1","s2","s3","s4","s5","t1","t2","t3","t4","t5"],[CTX]*10,PRED,[PRD]*5)
fig.legend(handles=[Patch(color=CONTENT,label="context (real)"),Patch(color=HISTC,label="history fill"),
                    Patch(color=PROMPT,label="trainable prompt"),Patch(color=PREDC,label="predicted token")],
           loc="upper center",ncol=4,fontsize=10,bbox_to_anchor=(0.5,1.05),frameon=False,handlelength=1.3,columnspacing=1.3)
out="/Users/zizhaohu/Desktop/projects/context-is-the-new-weight/.claude/worktrees/exp/paper/attention-sink/figures/windowed_attention.png"
plt.savefig(out,dpi=170,bbox_inches="tight"); print("wrote",out)
