"""Generate separate EN and ZH A-SYNC reports — clean, single-language PDFs.

Produces:
  docs/a_sync_report_en.pdf — English only (DejaVu fonts)
  docs/a_sync_report_zh.pdf — Chinese only (WenQuanYi Zen Hei fonts)
"""
import json, math, os
from datetime import datetime
import matplotlib, matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from fpdf import FPDF

matplotlib.use("Agg")
CHART_DIR = os.path.join("docs", "figures", "final_report")
os.makedirs(CHART_DIR, exist_ok=True)
plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.titlesize": 10,
                     "axes.labelsize": 8.5, "legend.fontsize": 7, "figure.figsize": (7.5, 4)})

C = {"blue": "#2563EB", "red": "#DC2626", "green": "#16A34A", "orange": "#EA580C",
     "purple": "#7C3AED", "gray": "#6B7280", "cyan": "#0891B2", "pink": "#DB2777",
     "dark": "#1F2937", "amber": "#D97706", "teal": "#0D9488"}

# ═══════════════════════════════════════════════════════════════════════
# CHARTS
# ═══════════════════════════════════════════════════════════════════════

def chart_residual():
    """Residual amplification theory + cross-depth benchmark (side-by-side)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 4))
    depths = np.arange(1, 37); rho = 1.08; amp = rho ** (depths - 1)
    ax1.fill_between(depths, 1, amp, alpha=0.06, color=C["red"])
    ax1.axvspan(1, 24, alpha=0.05, color=C["green"])
    ax1.axvspan(28, 36, alpha=0.05, color=C["red"])
    ax1.plot(depths, amp, color=C["red"], linewidth=2.5, zorder=3)
    pts = [(12,2.33,"OPT-125m"),(22,5.03,"TinyLlama"),(24,5.87,"Qwen0.5B"),(28,7.99,"Qwen7B")]
    for L, val, lbl in pts:
        ax1.scatter(L, val, color=C["green"] if L<28 else C["red"], s=80, zorder=4, edgecolors="white", lw=1.5)
        ax1.annotate(f"{lbl}\n{L}L x{val:.1f}", xy=(L,val), xytext=(L-3.5, val+0.7), fontsize=6.5, fontweight="bold",
                    arrowprops=dict(arrowstyle="->",color=C["dark"],lw=0.7),
                    bbox=dict(boxstyle="round,pad=0.2",fc="white",ec=C["gray"],alpha=0.8))
    ax1.axhline(0.005, color=C["gray"], ls=":", lw=0.8, alpha=0.4)
    ax1.text(3, 0.007, "SGD recovery/cycle (alpha*50)", fontsize=6, color=C["gray"])
    ax1.set_xlabel("Model Depth (layers)"); ax1.set_ylabel("Amplification Factor")
    ax1.set_title("Residual Amplification", fontweight="bold")
    ax1.set_yscale("log"); ax1.grid(True, alpha=0.15)
    leg = [mpatches.Patch(fc=C["green"],alpha=0.3,label="Stable"),mpatches.Patch(fc=C["red"],alpha=0.3,label="Divergent")]
    ax1.legend(handles=leg, fontsize=6.5, loc="upper left")

    md = json.load(open("runs/p1.2_depth/results.json"))["models"]
    for i,(m,col) in enumerate(zip(md,[C["green"]]*3+[C["red"]])):
        ppl = m['final_ppl']; s = f"PPL={ppl:.0f}" if not math.isinf(ppl) else "DIVERGED"
        ax2.bar(i,1,color=col,alpha=0.15,width=0.6)
        ax2.text(i,0.5,f"{m['model']}\n{m['layers']}L\n{s}",ha="center",va="center",fontsize=7,fontweight="bold",
                color="white",bbox=dict(boxstyle="round,pad=0.3",fc=col,alpha=0.85))
    ax2.axvline(2.5,color=C["red"],ls="--",lw=2,alpha=0.6)
    ax2.text(2.5,1.12,"Divergence Boundary",ha="center",fontsize=7,color=C["red"],fontweight="bold")
    ax2.set_title("Cross-Depth Benchmark",fontweight="bold"); ax2.set_ylim(0,1.2)
    ax2.set_yticks([]); ax2.set_xticks(range(4)); ax2.set_xticklabels(["12L","22L","24L","28L"])
    fig.suptitle("Why Protocol A Fails on Deep Models", fontweight="bold", fontsize=12)
    plt.tight_layout(); p=os.path.join(CHART_DIR,"fig1_residual.png"); fig.savefig(p,dpi=150,bbox_inches="tight"); plt.close(fig)
    return p

def chart_architecture():
    """Transformer architecture diagram with residual amplification path."""
    fig, ax = plt.subplots(figsize=(7.5,5)); ax.set_xlim(0,12); ax.set_ylim(-1,13); ax.axis("off")
    bx, bw, bh, gap = 1, 3.5, 1.7, 0.4
    for i in range(6):
        y = 11 - i*(bh+gap); rl = 28 if i==5 else (i*5+3 if i>0 else 1)
        col_bg = "#E8F5E9" if i<5 else "#FFEBEE"
        rect = mpatches.FancyBboxPatch((bx,y),bw,bh,boxstyle="round,pad=0.1",fc=col_bg,ec=C["gray"],lw=1,alpha=0.7)
        ax.add_patch(rect)
        ax.text(bx+0.2,y+bh-0.3,f"Block {rl}",fontsize=7.5,fontweight="bold",color=C["dark"])
        for j, (mn, fc) in enumerate([("QKV",C["blue"]),("O",C["blue"]),("Gate",C["teal"]),("Up",C["teal"]),("Down",C["teal"])]):
            sy = y+0.2+j*0.3; r2=mpatches.FancyBboxPatch((bx+0.3,sy),2.8,0.28,boxstyle="round,pad=0.05",fc=fc,alpha=0.25,ec=fc,lw=0.5)
            ax.add_patch(r2); ax.text(bx+0.4,sy+0.14,mn,fontsize=5.5,color=fc,fontweight="bold")
        if i<5:
            ax.annotate("",xy=(bx-0.3,y-0.4),xytext=(bx-0.3,y+bh+0.1),arrowprops=dict(arrowstyle="->",color=C["orange"],lw=1.5,connectionstyle="arc3,rad=0"))
            ax.text(bx-1.0,y+bh/2,f"x{1.08**(i):.1f}",fontsize=6,color=C["orange"],fontweight="bold",rotation=90,va="center")
    lmy=12.5; lm=mpatches.FancyBboxPatch((bx,lmy),bw,0.55,boxstyle="round,pad=0.1",fc=C["red"],alpha=0.2,ec=C["red"],lw=1.5)
    ax.add_patch(lm); ax.text(bx+bw/2,lmy+0.28,"lm_head",ha="center",fontsize=8,fontweight="bold",color=C["red"])
    ax.annotate("ALS modifies\nONLY lm_head",xy=(bx+bw/2,lmy),xytext=(bx+bw+1.5,lmy+0.3),fontsize=7.5,fontweight="bold",color=C["red"],arrowprops=dict(arrowstyle="->",color=C["red"],lw=2))
    ax.annotate("Perturbation propagates\nthrough L-1 frozen blocks",xy=(bx-0.3,5.5),xytext=(bx+bw+0.8,8),fontsize=6.5,color=C["orange"],arrowprops=dict(arrowstyle="->",color=C["orange"],lw=1.2,connectionstyle="arc3,rad=0.3"))
    ax.text(7,1.5,"Total amplification: 8.0x\nSGD recovery: 0.005x\nImbalance: 1600:1",fontsize=7.5,fontweight="bold",color=C["red"],bbox=dict(boxstyle="round,pad=0.4",fc="#FFEBEE",ec=C["red"]))
    ax.set_title("Residual Amplification Path in Protocol A",fontweight="bold",fontsize=11)
    plt.tight_layout(); p=os.path.join(CHART_DIR,"fig2_arch.png"); fig.savefig(p,dpi=150,bbox_inches="tight"); plt.close(fig)
    return p

def chart_matrix():
    """Algorithm module application matrix."""
    variants = ["Protocol A\n(original)","Depth\nProtection","LARS\nOptimizer","Gradient\nClipping",
               "Multi-layer\nALS Batch","Multi-layer\nALS Seq","A-CASCADE","A-RAPID","A-DUAL","A-KD","A-PROBE",
               "A-SYNC\n+perturb","A-SYNC\nno-perturb","A-SYNC\nCONSTANT"]
    modules = ["lm_head","Emb","Q","K","V","O_proj","Gate","Up","Down"]
    M = np.array([[3,0,0,0,0,0,0,0,0],[3,0,0,0,0,0,0,0,0],[1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1],
                  [3,0,2,2,2,2,2,2,2],[3,0,2,2,2,2,2,2,2],[3,0,0,0,0,0,0,0,0],[3,0,0,0,0,0,0,0,0],
                  [3,0,1,1,1,1,1,1,1],[3,0,1,1,1,1,1,1,1],[0,0,0,0,0,0,0,0,0],
                  [2,0,1,1,1,1,1,1,1],[2,0,1,1,1,1,1,1,1],[2,0,1,1,1,1,1,1,1]])
    cmap = np.full(M.shape+(3,), 255)
    for i in range(len(variants)):
        for j in range(len(modules)):
            if M[i,j]==0: cmap[i,j]=[220,220,220]
            elif M[i,j]==1: cmap[i,j]=[212,239,223]
            elif M[i,j]==2: cmap[i,j]=[254,235,201]
            elif M[i,j]==3: cmap[i,j]=[245,203,203]
    fig,ax=plt.subplots(figsize=(8,5.5)); ax.imshow(cmap/255.0,aspect="auto")
    ax.set_xticks(range(len(modules))); ax.set_xticklabels(modules,fontsize=7,rotation=45,ha="right")
    ax.set_yticks(range(len(variants))); ax.set_yticklabels(variants,fontsize=6)
    for i in range(len(variants)):
        if "SYNC" in variants[i]: ax.axhline(i-0.5,color=C["blue"],lw=1.5,alpha=0.3); ax.axhline(i+0.5,color=C["blue"],lw=1.5,alpha=0.3)
    for i in range(len(variants)):
        for j in range(len(modules)):
            if M[i,j]>0: ax.text(j,i,{1:"SGD",2:"+d",3:"ALS"}[M[i,j]],ha="center",va="center",fontsize=5.5,fontweight="bold")
    leg=[mpatches.Patch(fc=(245/255,203/255,203/255),label="ALS (full solve)"),mpatches.Patch(fc=(254/255,235/255,201/255),label="A-SYNC (gradient inject)"),mpatches.Patch(fc=(212/255,239/255,223/255),label="SGD (standard)"),mpatches.Patch(fc=(220/255,220/255,220/255),label="Untouched")]
    ax.legend(handles=leg,fontsize=6.5,loc="lower left",bbox_to_anchor=(1.02,0),ncol=1)
    ax.set_title("Algorithm Module Application Matrix",fontweight="bold",fontsize=10)
    plt.tight_layout(); p=os.path.join(CHART_DIR,"fig3_matrix.png"); fig.savefig(p,dpi=150,bbox_inches="tight"); plt.close(fig)
    return p

def chart_convergence():
    """Convergence curves on Qwen7B."""
    data={"48-constant (BEST)":{"d":json.load(open("runs/a_sync_48cycle_7b.json")),"c":C["blue"],"ls":"-","m":"o"},
          "24-constant":{"d":json.load(open("runs/a_sync_constant_7b.json")),"c":C["cyan"],"ls":"--","m":"s"},
          "16-cosine":{"d":json.load(open("runs/a_sync_swa_cosine_7b.json")),"c":C["purple"],"ls":"-.","m":"D"},
          "8 no-perturb":{"d":json.load(open("runs/a_sync_noperturb_8cycle_7b.json")),"c":C["green"],"ls":":","m":"^"},
          "Pure SGD":{"d":json.load(open("runs/sgd_vs_async_7b.json")),"c":C["red"],"ls":":","m":"x"}}
    fig,ax=plt.subplots(figsize=(8,4.5))
    for lb,dv in data.items():
        p=dv["d"].get("ppls",dv["d"].get("pure_sgd",{}).get("ppls",[])); xs=list(range(1,len(p)+1))
        ax.plot(xs,p,color=dv["c"],ls=dv["ls"],lw=1.8,marker=dv["m"],ms=4,markevery=6,label=lb,alpha=0.9)
    ax.axhline(73,color=C["gray"],ls=":",lw=0.8,alpha=0.3); ax.text(2,74,"Baseline PPL=73",fontsize=7,color=C["gray"])
    ax.axhline(10,color=C["red"],ls="--",lw=0.8,alpha=0.2); ax.text(44,11,"Original Protocol A: diverges",fontsize=7,color=C["red"],ha="right")
    ax.set_xlabel("Training Cycle"); ax.set_ylabel("Perplexity (lower=better)"); ax.set_title("A-SYNC Convergence on Qwen2.5-7B (28L)",fontweight="bold")
    ax.set_yscale("log"); ax.grid(True,alpha=0.15); ax.legend(fontsize=7,ncol=2)
    plt.tight_layout(); p=os.path.join(CHART_DIR,"fig4_convergence.png"); fig.savefig(p,dpi=150,bbox_inches="tight"); plt.close(fig)
    return p

def chart_lars():
    """LARS vs SGD comparison."""
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(8,3.5))
    for fname,ax,title in [("lars_sanity_gpt2.json",ax1,"GPT-2 125M (12L)"),("lars_qwen05b.json",ax2,"Qwen2.5-0.5B (24L)")]:
        d=json.load(open(f"runs/{fname}")); r=d["results"]
        for lb,clr,mkr in [("SGD",C["blue"],"o"),("LARS",C["orange"],"s")]:
            p=r[lb]["ppls"]; fin=[(i+1,x) for i,x in enumerate(p) if not math.isinf(x) and x<1e10]
            if fin: xi,yi=zip(*fin); ax.plot(xi,yi,color=clr,marker=mkr,lw=2,ms=6,label=lb)
        bl=d.get("baseline_ppl",0)
        if bl<1e6: ax.axhline(bl,color=C["gray"],ls=":",lw=0.8,alpha=0.4); ax.text(2,bl*1.1,"Baseline",fontsize=6.5,color=C["gray"])
        ax.set_title(title,fontweight="bold",fontsize=9); ax.set_yscale("log"); ax.grid(True,alpha=0.2); ax.legend(fontsize=7)
    fig.suptitle("LARS: Reduces NaN but Does Not Converge",fontweight="bold")
    plt.tight_layout(); p=os.path.join(CHART_DIR,"fig5_lars.png"); fig.savefig(p,dpi=150,bbox_inches="tight"); plt.close(fig)
    return p

def chart_depth():
    """Protocol A vs A-SYNC depth boundary."""
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(8,3.8))
    mdls=["OPT-125m","TinyLlama","Qwen0.5B","Qwen7B"]; dp=[12,22,24,28]
    for i,(m,l,lbl,col) in enumerate(zip(mdls,dp,["PPL=107","PPL=16","PPL=18","11/11 DIVERGED"],[C["green"]]*3+[C["red"]])):
        ax1.bar(i,1,color=col,alpha=0.15,width=0.6)
        ax1.text(i,0.5,f"{m}\n{l}L\n{lbl}",ha="center",va="center",fontsize=7,fontweight="bold",color="white",bbox=dict(boxstyle="round,pad=0.3",fc=col,alpha=0.85))
    ax1.axvline(2.5,color=C["red"],ls="--",lw=2); ax1.text(2.5,1.15,"FAILURE",ha="center",fontsize=7.5,color=C["red"],fontweight="bold")
    ax1.set_title("Protocol A (Original)",fontweight="bold"); ax1.set_ylim(0,1.3); ax1.set_yticks([]); ax1.set_xticks(range(4)); ax1.set_xticklabels([f"{d}L" for d in dp])
    for i,(m,l,lbl,col) in enumerate(zip(mdls,dp,["PPL~107","PPL~15","PPL 5.5","PPL 7.6"],[C["green"]]*3+[C["blue"]])):
        ax2.bar(i,1,color=col,alpha=0.15,width=0.6)
        ax2.text(i,0.5,f"{m}\n{l}L\n{lbl}",ha="center",va="center",fontsize=7,fontweight="bold",color="white",bbox=dict(boxstyle="round,pad=0.3",fc=col,alpha=0.85))
    ax2.axvline(2.5,color=C["blue"],ls="-",lw=2); ax2.text(2.5,1.15,"CROSSED!",ha="center",fontsize=7.5,color=C["blue"],fontweight="bold")
    ax2.set_title("A-SYNC (Ours)",fontweight="bold"); ax2.set_ylim(0,1.3); ax2.set_yticks([]); ax2.set_xticks(range(4)); ax2.set_xticklabels([f"{d}L" for d in dp])
    fig.suptitle("Depth Boundary: Before vs After",fontweight="bold",fontsize=12)
    plt.tight_layout(); p=os.path.join(CHART_DIR,"fig6_depth.png"); fig.savefig(p,dpi=150,bbox_inches="tight"); plt.close(fig)
    return p

def chart_scoreboard():
    """All fix attempts scoreboard."""
    att=[("A-SYNC 48-constant",7.6,C["blue"]),("A-SYNC 16-cosine",10.5,C["cyan"]),("A-SYNC 8 no-perturb",16.6,C["green"]),
         ("A-PROBE",22.8,C["teal"]),("Pure SGD",22.5,C["gray"]),("LARS",161674,C["orange"]),
         ("Multi-layer ALS",1e8,C["red"]),("A-CASCADE",1e20,C["red"]),("A-RAPID",1e28,C["red"]),
         ("A-KD",195,C["pink"]),("A-DUAL",1e10,C["red"]),("Parameter tuning",1e10,C["red"])]
    fig,ax=plt.subplots(figsize=(8,5)); nm,vs,cs=zip(*att)
    bars=ax.barh(range(len(nm)),vs,color=cs,alpha=0.85,height=0.5)
    for bar,val in zip(bars,vs): ax.text(bar.get_width()+0.3,bar.get_y()+bar.get_height()/2,f"PPL {val:.0f}" if val<1e6 else "DIVERGE",va="center",fontsize=7,fontweight="bold" if val<20 else "normal")
    ax.set_yticks(range(len(nm))); ax.set_yticklabels(nm,fontsize=7); ax.set_xlabel("Final PPL (log)"); ax.set_xscale("log")
    ax.set_title("All Fix Attempts — Qwen2.5-7B (28L)",fontweight="bold"); ax.invert_yaxis(); ax.grid(True,alpha=0.2,axis="x")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout(); p=os.path.join(CHART_DIR,"fig7_scoreboard.png"); fig.savefig(p,dpi=150,bbox_inches="tight"); plt.close(fig)
    return p

# ═══════════════════════════════════════════════════════════════════════
# PDF BASE
# ═══════════════════════════════════════════════════════════════════════

class PDF(FPDF):
    def __init__(self, lang="en"):
        super().__init__("P","mm","A4"); self.set_auto_page_break(True,20); self.lang = lang
        if lang == "en":
            self.add_font("F","","/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
            self.add_font("F","B","/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
            self.add_font("F","I","/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf")
            self.add_font("M","","/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")
        else:
            self.add_font("F","","/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
            self.add_font("F","B","/usr/share/fonts/truetype/wqy/wqy-microhei.ttc")
            self.add_font("F","I","/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")  # fallback
            self.add_font("M","","/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")

    def header(self):
        if self.page_no()==1: return
        self.set_font("F","I",6.5); self.set_text_color(130,130,130)
        t = "Protocol A-SYNC Report" if self.lang=="en" else "Protocol A-SYNC 报告"
        self.cell(0,4.5,t,align="L"); self.cell(0,4.5,f"Page {self.page_no()}",align="R",new_x="LMARGIN",new_y="NEXT")
        self.line(self.l_margin,self.get_y(),self.w-self.r_margin,self.get_y()); self.ln(2.5)

    def footer(self):
        self.set_y(-14); self.set_font("F","I",5.5); self.set_text_color(150,150,150)
        self.cell(0,7,f"Generated {datetime.now().strftime('%Y-%m-%d')} | alternating-optimization-lora",align="C")

    def stitle(self):
        self.add_page(); self.ln(30)
        self.set_font("F","B",22); self.set_text_color(*self._rgb(C["dark"]))
        tl = "Protocol A-SYNC\nFrom Divergence to Convergence\non Deep Models" if self.lang=="en" else "Protocol A-SYNC\n从发散到收敛\n深层模型优化报告"
        self.multi_cell(0,10,tl,align="C"); self.ln(4)
        self.set_font("F","",11); self.set_text_color(*self._rgb(C["gray"]))
        self.cell(0,7,"Algorithm Variant Report / July 2026" if self.lang=="en" else "算法变体报告 / 2026年7月",align="C",new_x="LMARGIN",new_y="NEXT"); self.ln(10)
        y0=self.get_y(); self.set_draw_color(*self._rgb(C["blue"])); self.rect(25,y0,self.w-50,42,style="D")
        self.set_fill_color(*self._rgb(C["blue"])); self.set_text_color(255,255,255); self.set_xy(25,y0+2)
        self.set_font("F","B",10); self.cell(self.w-50,7,"  KEY RESULT" if self.lang=="en" else "  核心成果",align="C")
        self.set_xy(25,y0+11); self.set_text_color(40,40,40); self.set_font("F","",8)
        if self.lang=="en":
            self.multi_cell(self.w-50,4.5,"Protocol A diverged on all 28+ layer models.\nA-SYNC replaces direct weight writes with gradient injection.\nQwen2.5-7B (28L): PPL 58.8 -> 7.6 in 48 cycles. Monotonic.\n\nAll 7 prior fix attempts (tuning, LARS, clipping, multi-layer ALS, etc.) failed.",align="C")
        else:
            self.multi_cell(self.w-50,4.5,"Protocol A 在 28+ 层模型上全部发散。\nA-SYNC 用梯度注入替代直接权重写入。\nQwen2.5-7B（28L）：PPL 58.8 -> 7.6，48周期单调收敛。\n\n此前 7 种修复尝试（调参、LARS、裁剪、多层 ALS 等）全部失败。",align="C")

    def sec(self, title): self.ln(3); self.set_font("F","B",11); self.set_text_color(*self._rgb(C["dark"])); self.cell(0,6.5,title,new_x="LMARGIN",new_y="NEXT"); self.set_draw_color(*self._rgb(C["blue"])); self.set_line_width(0.4); self.line(self.l_margin,self.get_y(),self.w-self.r_margin,self.get_y()); self.ln(2.5)
    def body(self,t): self.set_font("F","",8); self.set_text_color(50,50,50); self.multi_cell(self.w-2*self.l_margin,4.3,t)
    def bold(self,t): self.set_font("F","B",8); self.set_text_color(50,50,50); self.multi_cell(self.w-2*self.l_margin,4.3,t)
    def code(self,t):
        self.set_font("M","",6.5); self.set_text_color(60,60,60); self.set_fill_color(248,248,248)
        for ln in t.split("\n"): self.cell(0,3.5,f"  {ln}",fill=True,new_x="LMARGIN",new_y="NEXT")
    def tbl(self,h,rows,cw=None):
        if cw is None: cw=[(self.w-2*self.l_margin)/len(h)]*len(h)
        self.set_font("F","B",6.5); self.set_fill_color(*self._rgb(C["dark"])); self.set_text_color(255,255,255)
        for hh,w in zip(h,cw): self.cell(w,5.5,f" {hh}",fill=True,border=0); self.ln()
        self.set_text_color(50,50,50)
        for i,row in enumerate(rows):
            self.set_font("F","",6.5); bg=(248,248,248)if i%2==0 else(255,255,255); self.set_fill_color(*bg)
            for c,w in zip(row,cw): self.cell(w,5,f" {c}",fill=True,border=0); self.ln()
        self.ln(2)
    def img(self,path,w=175):
        if os.path.exists(path): self.image(path,x=(self.w-w)/2,w=w); self.ln(2)
        else: self.body(f"[Image missing: {path}]")
    def callout(self,t,clr=C["red"]):
        r,g,b=self._rgb(clr); self.set_fill_color(r,g,b); self.set_text_color(255,255,255); self.set_font("F","B",7.5)
        self.cell(self.w-2*self.l_margin,5,f"  {t}",fill=True,new_x="LMARGIN",new_y="NEXT"); self.ln(2)
    @staticmethod
    def _rgb(h): h = h.lstrip("#"); return tuple(int(h[i:i+2],16) for i in(0,2,4))

# ═══════════════════════════════════════════════════════════════════════
# ENGLISH REPORT
# ═══════════════════════════════════════════════════════════════════════

def build_en(paths):
    pdf=PDF("en"); pdf.stitle()

    pdf.sec("1. Motivation: Why Original Protocol A Fails")
    pdf.body("Protocol A interleaves three phases: ALS (exact block-wise least squares on lm_head), SGD (stochastic gradient descent on all parameters), and Perturb (random noise injection). On models with 12-24 transformer layers, this converges reliably. On models with 28+ layers, every attempt diverges within 2-3 cycles.")
    pdf.ln(1)

    pdf.bold("Root Cause: Residual Amplification")
    pdf.body("ALS modifies only the lm_head (output projection layer). The perturbation propagates forward through L-1 frozen transformer blocks via residual connections x + sublayer(x). Each residual hop amplifies by rho = 1.08. After 27 connections in a 28-layer Qwen2.5-7B, amplification is rho^27 = 8.0x. SGD recovers at most 0.005 per cycle — a 1600:1 asymmetry causing catastrophic divergence.")
    pdf.ln(2)

    pdf.img(paths["residual"])
    pdf.set_font("F","I",6); pdf.set_text_color(120,120,120)
    pdf.cell(0,3,"Fig 1: Residual amplification rho^(L-1) vs model depth (left). Cross-depth benchmark (right).",align="C",new_x="LMARGIN",new_y="NEXT")
    pdf.ln(2)

    pdf.tbl(["Model","Layers","Amplification","Protocol A PPL","Status"],
            [["OPT-125m","12","x2.3","106.9","SUCCESS"],["TinyLlama-1.1B","22","x5.0","15.5","SUCCESS"],
             ["Qwen2.5-0.5B","24","x5.9","18.0","SUCCESS (unstable)"],["Qwen2.5-7B","28","x8.0","DIVERGED","11/11 FAIL"]],
            [35,20,25,28,30])

    pdf.img(paths["arch"])
    pdf.set_font("F","I",6); pdf.set_text_color(120,120,120)
    pdf.cell(0,3,"Fig 2: Transformer architecture — ALS modifies only lm_head. Perturbation propagates through L-1 frozen blocks.",align="C",new_x="LMARGIN",new_y="NEXT")
    pdf.ln(2)
    pdf.callout("The problem is structural, not parametric. ALS's lm_head-only design creates a 1600:1 amplification-to-recovery asymmetry in 28-layer models.",C["red"])

    pdf.add_page()
    pdf.sec("2. A-SYNC Algorithm: Gradient Injection")
    pdf.bold("Core Innovation")
    pdf.body("Instead of directly writing the ALS-optimized weight into lm_head, A-SYNC computes the delta dW = W_new - W_old, reverts the weight, and injects the delta as a gradient bias during each SGD step. This allows head and body to co-evolve: the ALS direction guides SGD without creating the frozen-body amplification chain.")
    pdf.ln(2)
    pdf.bold("One Cycle Pseudocode:")
    pdf.code("1. ALS solve on lm_head -> get W_new (label-based exact least squares)\n2. Compute delta = W_new - W_old (CPU offload to save GPU memory)\n3. Revert lm_head to W_old\n4. SGD 50 steps: each step add sync_strength * delta to lm_head gradient\n5. (Perturbation: REMOVED — causes oscillations)\n6. Repeat from step 1")
    pdf.ln(2)
    pdf.bold("Final Configuration:")
    pdf.code("sync_strength: 0.05 (CONSTANT, no decay)\nlearning_rate: 2e-4 (CONSTANT)\nmomentum: 0.0, weight_decay: 0.01\ncycles: 24-48 (converges at ~44)\nALS: block_size=512, reg_lambda=1e-3, step_size=0.01\nPerturbation: DISABLED")
    pdf.ln(2)
    pdf.callout("A-SYNC: ALS computes WHERE to go (direction). SGD handles HOW to get there (optimization). The two signals are orthogonal (cos ~ 0) and complementary.",C["blue"])
    pdf.ln(2)
    pdf.img(paths["matrix"])
    pdf.set_font("F","I",6); pdf.set_text_color(120,120,120)
    pdf.cell(0,3,"Fig 3: Algorithm Module Application Matrix — which variant touches which parameters. A-SYNC (bottom rows) only gradient-injects lm_head.",align="C",new_x="LMARGIN",new_y="NEXT")

    pdf.add_page()
    pdf.sec("3. All Fix Attempts Explained")
    attempts = [
        ("Parameter Tuning","Failed","Reducing alpha to 0.001 requires 10x steps; tuning ALS:SGD ratio from 1:5 to 1:50 found no stabilization for 28L. The problem is structural, not parametric."),
        ("Depth-Boundary Protection","Partial","skip_early_ratio (skip first 50% layers) + depth_decay_beta (exponential damping) + clip_catastrophic. Extends stable regime from 12L to 24L. At 28L, clip aborts all meaningful updates."),
        ("LARS Optimizer","Failed","Layer-wise Adaptive Rate Scaling. GPT-2: SGD PPL 88>18, LARS PPL 173>146 (stagnates). Qwen0.5B: avoids NaN but PPL=161k (no convergence)."),
        ("Multi-Layer ALS","Failed","Batch: stale activations from simultaneous modifications > instant divergence. Sequential: intermediate-layer ALS uses self-reconstruction X*W_old^T, X^T*X is 4864x4864 rank<=256 > underdetermined, solution is noise."),
        ("A-PROBE","Partial","Rank-64 probe before lm_head. ALS solves 64x64 Cholesky (trivial). Proves residual amplification can be eliminated architecturally, but rank bottleneck caps performance at pure SGD level (PPL 22.8 on 7B)."),
        ("A-SYNC (Gradient Injection)","SUCCESS","ALS computes optimal lm_head delta direction, injects as gradient bias in SGD. Head and body co-evolve. ALS delta is orthogonal to SGD gradient (cos~0). Constant sync=0.05 dominates all decay schedules. Qwen7B PPL 58.8>7.6 — first Protocol A variant to converge on deep models."),
    ]
    for name, result, desc in attempts:
        rcol = C["red"] if result=="Failed" else (C["green"] if result=="SUCCESS" else C["orange"])
        pdf.bold(f"{name} — {result}")
        pdf.body(desc); pdf.ln(1.5)

    pdf.img(paths["lars"])
    pdf.set_font("F","I",6); pdf.set_text_color(120,120,120)
    pdf.cell(0,3,"Fig 4: LARS vs SGD on GPT-2 12L (left) and Qwen0.5B 24L (right). LARS avoids NaN but does not converge.",align="C",new_x="LMARGIN",new_y="NEXT")

    pdf.add_page()
    pdf.sec("4. A-SYNC Variant Progression")
    pdf.body("The path from the first A-SYNC to the final converging variant:")
    pdf.ln(2)
    pdf.tbl(["#","Variant","Key Change","7B Final PPL","Status"],
            [["1","A-SYNC +perturb","Gradient inject + noise","25.8","Oscillates"],["2","A-SYNC no-perturb","Remove perturbation","16.6","Monotonic"],
             ["3","A-SYNC 16-cosine","Cosine sync decay","10.5","Good, sync dies early"],["4","A-SYNC 32-cosine","Cosine over 32 cycles","13.3","Decay kills tail"],
             ["5","A-CYCLE restart","3x8 warm restart","16.5","Window too short"],["6","A-SYNC 24-const","Constant sync, 24c","9.0","Excellent"],
             ["7","A-SYNC 48-const","Constant sync, 48c","7.6","BEST — converged"]],
            [8,36,42,28,28])
    pdf.ln(1)
    pdf.img(paths["convergence"])
    pdf.set_font("F","I",6); pdf.set_text_color(120,120,120)
    pdf.cell(0,3,"Fig 5: A-SYNC convergence curves on Qwen2.5-7B (28L). 48-cycle constant sync (blue) is the clear winner.",align="C",new_x="LMARGIN",new_y="NEXT")
    pdf.ln(2)
    pdf.callout("Key discovery: constant sync strength (0.05) dominates all decay schedules. The ALS signal should persist at full strength — it is guiding, not annealing.",C["blue"])

    pdf.ln(2)
    pdf.sec("5. Final Scoreboard")
    pdf.tbl(["Approach","Category","Qwen7B (28L) PPL","Status"],
            [["A-SYNC 48-constant","Gradient Injection","7.6","CONVERGED"],
             ["Pure SGD","Baseline","22.5","Plateaus"],["A-PROBE","Architecture","22.8","Bottleneck limited"],
             ["A-KD","Distillation","195","KL explosion"],["LARS","Optimizer","161k","No convergence"],
             ["Multi-layer ALS","Algorithm","DIVERGED","Underdetermined"],["Parameter Tuning","Tuning","DIVERGED","Non-transferable"],
             ["Protocol A (original)","Baseline","DIVERGED","Residual amplification"]],
            [42,28,32,38])
    pdf.ln(2)
    pdf.img(paths["scoreboard"])
    pdf.set_font("F","I",6); pdf.set_text_color(120,120,120)
    pdf.cell(0,3,"Fig 6: All fix attempts scoreboard — Qwen2.5-7B (28L). A-SYNC is 3x better than pure SGD.",align="C",new_x="LMARGIN",new_y="NEXT")
    pdf.ln(2)
    pdf.img(paths["depth"])
    pdf.set_font("F","I",6); pdf.set_text_color(120,120,120)
    pdf.cell(0,3,"Fig 7: Protocol A (left) diverges at 28L. A-SYNC (right) crosses the boundary — monotonic convergence at all depths.",align="C",new_x="LMARGIN",new_y="NEXT")

    out = os.path.join("docs","a_sync_report_en.pdf"); pdf.output(out); return out

# ═══════════════════════════════════════════════════════════════════════
# CHINESE REPORT
# ═══════════════════════════════════════════════════════════════════════

def build_zh(paths):
    pdf=PDF("zh"); pdf.stitle()

    pdf.sec("1. 动机：为什么原始 Protocol A 失败")
    pdf.body("Protocol A 交替执行三个阶段：ALS（对 lm_head 精确逐块最小二乘求解）、SGD（对所有参数随机梯度下降）和 Perturb（随机噪声注入）。在 12-24 层 Transformer 模型上可靠收敛。在 28 层及以上模型上，所有尝试均在 2-3 周期内发散。")
    pdf.ln(1)

    pdf.bold("根本原因：残差放大效应")
    pdf.body("ALS 仅修改 lm_head（输出投影层）。扰动通过残差连接 x + sublayer(x) 向前传播经过 L-1 个冻结的 Transformer 块。每次跳跃放大约 rho=1.08 倍。经过 27 次连接后，有效放大为 rho^27 = 8.0 倍。SGD 阶段每周期最多恢复 0.005，造成 1600:1 的不对称，导致灾难性发散。")
    pdf.ln(2)

    pdf.img(paths["residual"])
    pdf.set_font("F","I",6); pdf.set_text_color(120,120,120)
    pdf.cell(0,3,"图1：左——残差放大曲线 rho^(L-1) 与实验数据；右——跨深度基准测试结果。",align="C",new_x="LMARGIN",new_y="NEXT")
    pdf.ln(2)

    pdf.tbl(["模型","层数","放大倍数","Protocol A PPL","状态"],
            [["OPT-125m","12","2.3x","106.9","成功"],["TinyLlama-1.1B","22","5.0x","15.5","成功"],
             ["Qwen2.5-0.5B","24","5.9x","18.0","成功（不稳定）"],["Qwen2.5-7B","28","8.0x","发散","11/11 失败"]],
            [35,18,22,28,30])

    pdf.img(paths["arch"])
    pdf.set_font("F","I",6); pdf.set_text_color(120,120,120)
    pdf.cell(0,3,"图2：Transformer 架构——ALS 仅修改 lm_head，扰动通过 L-1 个冻结块传播。",align="C",new_x="LMARGIN",new_y="NEXT")
    pdf.ln(2)
    pdf.callout("问题的根源是结构性的，而非参数性的。ALS 仅修改 lm_head 的设计，在 28 层模型中造成了 1600:1 的放大-恢复不对称。",C["red"])

    pdf.add_page()
    pdf.sec("2. A-SYNC 算法：梯度注入")
    pdf.bold("核心创新")
    pdf.body("A-SYNC 不直接将 ALS 优化后的权重写入 lm_head，而是计算 delta dW = W_new - W_old，还原权重，并将 delta 作为梯度偏置在 SGD 每一步中注入。这使得头和体可以共同演化：ALS 方向引导 SGD，而不产生冻结体放大链。")
    pdf.ln(2)
    pdf.bold("一个周期的伪代码：")
    pdf.code("1. ALS 求解 lm_head -> 得到 W_new（基于标签的精确最小二乘）\n2. 计算 delta = W_new - W_old（CPU 卸放以节省 GPU 内存）\n3. 将 lm_head 还原为 W_old\n4. SGD 运行 50 步：每步将 sync_strength * delta 添加到 lm_head 梯度\n5.（Perturb 阶段：已移除——会导致振荡）\n6. 从步骤 1 重复")
    pdf.ln(2)
    pdf.bold("最终配置：")
    pdf.code("sync_strength: 0.05（恒定，不衰减）\nlearning_rate: 2e-4（恒定）\nmomentum: 0.0, weight_decay: 0.01\ncycles: 24-48（约第 44 周期收敛）\nALS: block_size=512, reg_lambda=1e-3, step_size=0.01\nPerturb 阶段：已禁用")
    pdf.ln(2)
    pdf.callout("A-SYNC 的设计哲学：ALS 计算该往哪里走（方向），SGD 负责如何走（优化）。两个信号正交（cos~0）且互补。",C["blue"])
    pdf.ln(2)
    pdf.img(paths["matrix"])
    pdf.set_font("F","I",6); pdf.set_text_color(120,120,120)
    pdf.cell(0,3,"图3：算法模块应用矩阵——各变体修改哪些参数。A-SYNC（底部行）仅对 lm_head 进行梯度注入。",align="C",new_x="LMARGIN",new_y="NEXT")

    pdf.add_page()
    pdf.sec("3. 所有修复尝试详述")
    attempts_zh = [
        ("参数调优（alpha 和 ALS:SGD 比值）","失败","将 alpha 从 0.01 降至 0.001 需要 10 倍训练时间；ALS:SGD 比值从 1:5 调至 1:50，28 层模型无论比值如何均发散。问题是结构性的，不是参数性的。"),
        ("深度边界保护","部分成功","三重保护：skip_early_ratio（跳过前 50% 层）、depth_decay_beta（指数衰减阻尼）、clip_catastrophic（回滚极端变化）。将稳定区间从 12 层扩展到 24 层。在 28 层，每周期触发 6-8 层裁剪，中止所有有意义的更新。"),
        ("LARS 优化器","失败","逐层自适应学习率。GPT-2 上：SGD PPL 88->18（收敛），LARS PPL 173->146（停滞）。Qwen0.5B 上：避免 NaN，但 PPL=161k（未收敛）。"),
        ("多层 ALS（批量+顺序）","失败","批量：一次前向传播同时修改多层 > 激活值过期 > 瞬间发散。顺序：逐层前向保证激活正确，但中间层使用自重建目标（X*W_old^T），X^T*X 为 4864x4864 但 batch_size=2 时秩<=256 > 严重欠定，解为噪声。"),
        ("A-PROBE（低秩瓶颈）","部分成功","在 lm_head 前插入秩为 64 的探针（3584->64->3584）。ALS 求解 64x64 Cholesky（极其轻量）。证明残差放大可通过架构设计消除，但秩瓶颈将性能限制在纯 SGD 水平（7B 上 PPL 22.8）。"),
        ("A-SYNC（梯度注入）","成功","ALS 计算最优 lm_head delta 方向，作为梯度偏置注入 SGD。头与体共同演化。ALS delta 与 SGD 梯度正交（cos~0），注入了一个 SGD 永远不会探索的方向。恒定 sync=0.05 支配所有衰减策略。Qwen7B 上 PPL 58.8->7.6——首个在深层模型上收敛的 Protocol A 变体。"),
    ]
    for name, result, desc in attempts_zh:
        pdf.bold(f"{name} —— {result}")
        pdf.body(desc); pdf.ln(1.5)

    pdf.img(paths["lars"])
    pdf.set_font("F","I",6); pdf.set_text_color(120,120,120)
    pdf.cell(0,3,"图4：LARS vs SGD——GPT-2 12L（左）和 Qwen0.5B 24L（右）。LARS 避免 NaN 但不收敛。",align="C",new_x="LMARGIN",new_y="NEXT")

    pdf.add_page()
    pdf.sec("4. A-SYNC 变体演进")
    pdf.body("从第一个 A-SYNC 到最终收敛变体的演进路径：")
    pdf.ln(2)
    pdf.tbl(["#","变体","关键改动","7B 最终 PPL","状态"],
            [["1","A-SYNC +perturb","梯度注入 + 噪声","25.8","振荡"],["2","A-SYNC no-perturb","移除扰动阶段","16.6","单调收敛"],
             ["3","A-SYNC 16-cosine","余弦衰减","10.5","衰减过早"],["4","A-SYNC 32-cosine","32周期余弦","13.3","衰减杀尾"],
             ["5","A-CYCLE restart","3x8 暖重启","16.5","窗口过短"],["6","A-SYNC 24-const","恒定同步 24c","9.0","优秀"],
             ["7","A-SYNC 48-const","恒定同步 48c","7.6","最佳——已收敛"]],
            [8,36,42,28,28])
    pdf.ln(1)
    pdf.img(paths["convergence"])
    pdf.set_font("F","I",6); pdf.set_text_color(120,120,120)
    pdf.cell(0,3,"图5：A-SYNC 在 Qwen2.5-7B（28L）上的收敛曲线。48-constant（蓝色）是明显的赢家。",align="C",new_x="LMARGIN",new_y="NEXT")
    pdf.ln(2)
    pdf.callout("关键发现：恒定的同步强度（0.05）支配所有衰减策略。ALS 信号应保持全强度——它是引导，不是退火。",C["blue"])

    pdf.ln(2)
    pdf.sec("5. 最终积分榜")
    pdf.tbl(["方法","类别","Qwen7B (28L) PPL","状态"],
            [["A-SYNC 48-constant","梯度注入","7.6","已收敛"],
             ["Pure SGD","基线","22.5","平台"],["A-PROBE","架构","22.8","瓶颈限制"],
             ["A-KD","知识蒸馏","195","KL 爆炸"],["LARS","优化器","161k","未收敛"],
             ["多层 ALS","算法","发散","欠定"],["参数调优","调参","发散","不可迁移"],
             ["Protocol A（原始）","基线","发散","残差放大"]],
            [42,28,32,38])
    pdf.ln(2)
    pdf.img(paths["scoreboard"])
    pdf.set_font("F","I",6); pdf.set_text_color(120,120,120)
    pdf.cell(0,3,"图6：所有修复尝试积分榜——Qwen2.5-7B（28L）。A-SYNC 比纯 SGD 好 3 倍。",align="C",new_x="LMARGIN",new_y="NEXT")
    pdf.ln(2)
    pdf.img(paths["depth"])
    pdf.set_font("F","I",6); pdf.set_text_color(120,120,120)
    pdf.cell(0,3,"图7：Protocol A（左）在 28L 发散。A-SYNC（右）跨越边界——在所有深度上单调收敛。",align="C",new_x="LMARGIN",new_y="NEXT")

    out = os.path.join("docs","a_sync_report_zh.pdf"); pdf.output(out); return out

# ═══════════════════════════════════════════════════════════════════════

def main():
    print("Generating 5 charts (shared)...")
    paths = {}
    paths["residual"] = chart_residual()
    paths["arch"] = chart_architecture()
    paths["matrix"] = chart_matrix()
    paths["convergence"] = chart_convergence()
    paths["lars"] = chart_lars()
    paths["depth"] = chart_depth()
    paths["scoreboard"] = chart_scoreboard()
    for n,p in paths.items(): print(f"  {n}: {p}")

    print("\nBuilding English PDF...")
    en_pdf = build_en(paths)
    print(f"  {en_pdf} ({os.path.getsize(en_pdf)/1024:.0f} KB)")

    print("Building Chinese PDF...")
    zh_pdf = build_zh(paths)
    print(f"  {zh_pdf} ({os.path.getsize(zh_pdf)/1024:.0f} KB)")

if __name__ == "__main__":
    main()
