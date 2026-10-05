import numpy as np
from quadruped_locomotion import QuadrupedGameEnv
env=QuadrupedGameEnv(seed=1)
o,i=env.reset(seed=1)
print("after reset: health",i["health"],"score",i["score"],"z",float(env.data.qpos[env.root_qpos_adr+2]))
lens=[]
for ep in range(5):
    env.reset(seed=100+ep)
    n=0
    while True:
        o,r,te,tr,info=env.step(np.zeros(12,dtype=np.float32))
        n+=1
        if te or tr:
            lens.append(n); print(f"ep{ep} end n={n} term={te} trunc={tr} health={info['health']} falls={info['falls']} z={float(env.data.qpos[env.root_qpos_adr+2]):.2f}")
            break
        if n>3000: lens.append(n); print("ep too long"); break
print("lens",lens)
env.close()
