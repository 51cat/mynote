# 服务器部署jupyter
1. Add conda path
```
export PATH="/SGRNJ03/randd/public/soft/conda/miniconda/bin:$PATH" 
 source ~/.bashrc`
```
```
#安装jupyter
# Assuming your conda-env is named cenv
conda create -n cenv
conda activate cenv
conda install -c anaconda jupyter
```

1. Install python kernel
```
conda install ipykernel
ipython kernel install --user --name=<any_name_for_kernel>
```

1. Install R kernel
```
conda install -c r r-irkernel
IRkernel::installspec(name="<any_name_for_kernel>",displayname="<any_name_for_kernel>")
```
4.Run
```
mkdir ~/.jupyter/
cp /SGRNJ/Database/script/pipe/develop/shared/config/jupyter_notebook_config.py ~/.jupyter/
mkdir /SGRNJ03/randd/user/{user}/jupyter && cd /SGRNJ03/randd/user/{user}/jupyter
nohup jupyter notebook &
```
5.updateR
conda install -c r r-base=4.0.3
 
