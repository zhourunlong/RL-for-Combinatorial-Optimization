import os

dirs = os.listdir("./")
for dir in dirs:
    if "Exp" not in dir:
        continue
    for par in ["checkpoint", "result"]:
        for sub in ["warmup", "final"]:
            ckptdir = os.path.join(dir, par, sub)
            ckpts = os.listdir(ckptdir)
            ckpts.sort()
            ckpts = ckpts[:-1]
            for ckpt in ckpts:
                os.remove(os.path.join(ckptdir, ckpt))
