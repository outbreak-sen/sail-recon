# 环境配置
sail-recon

~~~
git clone https://github.com/HKUST-SAIL/sail-recon.git
cd sail-recon
conda env create -n sail-recon python==3.11
conda activate sail-recon
pip install -e .

~~~
# 代码使用
无法完整输出tum数据集，需要采样
~~~
CUDA_VISIBLE_DEVICES=1 python demo_traj.py --img_dir /share/datasets/TUM/Dynamics/rgbd_dataset_freiburg2_desk_with_person/rgb --frame_interval 30 --max_frames 30 --out_dir outputs --batch_size 4 --debug_dump
~~~