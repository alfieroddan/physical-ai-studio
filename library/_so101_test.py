import torch
from physicalai.data import Feature, FeatureType, Observation
from physicalai.policies import MolmoAct2
dev="cuda"
inf=[Feature(name="overview",ftype=FeatureType.VISUAL,shape=(3,256,256)),Feature(name="wrist",ftype=FeatureType.VISUAL,shape=(3,256,256)),Feature(name="state",ftype=FeatureType.STATE,shape=(6,))]
outf=[Feature(name="action",ftype=FeatureType.ACTION,shape=(6,))]
def run(flag):
    torch.manual_seed(0)
    p=MolmoAct2(input_features=inf,output_features=outf,repo_id="molmo-LIBERO",n_obs_steps=1,n_action_steps=10,adapt_to_so101=flag)
    print("config.adapt_to_so101:",p.config.adapt_to_so101,"| post has transform:",p._postprocessor._joint_transform is not None)
    p=p.to(device=dev,dtype=torch.bfloat16); p.eval()
    b=Observation(images={"overview":torch.rand(1,3,256,256),"wrist":torch.rand(1,3,256,256)},state=torch.rand(1,6)*100,task=["pick up the cube"]).to(dev)
    with torch.no_grad():
        return p.predict_action_chunk(b)
a_off=run(False)
a_on=run(True)
print("off shape:",tuple(a_off.shape)," on shape:",tuple(a_on.shape))
print("outputs differ (transform applied):", not torch.allclose(a_off,a_on))
