#$-wd /vols/dune/zz5617/logs
#$-q gpu.q -l h_rt=3:0:0 
#$-m ea -M zz5617@ic.ac.uk
cd /vols/dune/zz5617/GAN-Challenge
python3 main.py ${SGE_TASK_ID}
