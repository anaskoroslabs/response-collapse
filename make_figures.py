import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np, json, os
os.makedirs("../figs", exist_ok=True)
plt.rcParams.update({'font.family':'serif','font.size':10,'mathtext.fontset':'cm'})
d=json.load(open("results/results.json"))
e=d["E1_decay_laws"]; T=np.array(e["T"])
fig,ax=plt.subplots(figsize=(3.3,2.7))
for key,c,lab in [("full","#B2182B",r"full predictive $L^2$"),
                  ("pointwise","#2166AC",r"pointwise $Q(y_0)$"),
                  ("mean","#4DAC26",r"smooth global $\int aQ$")]:
    y=np.array(e[key]); ax.loglog(T,y,'o',color=c,ms=4)
    fit=np.polyfit(np.log10(T),np.log10(y),1)
    ax.loglog(T,10**np.polyval(fit,np.log10(T)),'-',color=c,lw=1.6,
              label=lab+f"  ({round(fit[0],3)})")
ax.set_xlabel(r"$T$ (observations)"); ax.set_ylabel(r"$\chi_T^{\mathrm{do}}$")
ax.legend(fontsize=6.5,frameon=False,loc='lower left'); ax.grid(True,which='major',ls=':',alpha=.4)
plt.tight_layout(); plt.savefig("../figs/fig_decay.pdf"); plt.close()
e4=d["E4_correction_cost"]
fig,ax=plt.subplots(figsize=(3.3,2.5))
chans=["pointwise\npatch","distribution\n(field)","committed\nscalar"]
slopes=[e4["slope_pointwise"],e4["slope_field"],e4["slope_scalar"]]
ax.bar(chans,slopes,color=["#2166AC","#B2182B","#4DAC26"],alpha=.85)
for i,s in enumerate(slopes): ax.text(i,s+0.02,f"$T^{{{s}}}$",ha='center',fontsize=9)
ax.set_ylabel("correction cost exponent"); ax.set_ylim(0,1.15)
ax.set_title("aimed-correction cost = inverse channel leverage",fontsize=8)
plt.tight_layout(); plt.savefig("../figs/fig_correction.pdf"); plt.close()
print("figures written to ../figs/")
