# python3 diffusion_itm.py --task vismin_relation --version 3-m
# python3 diffusion_itm.py --task vismin_relation -—img_retrieval --version 3-m
# python3 diffusion_itm.py --task vismin_attribute --version 3-m
# python3 diffusion_itm.py --task vismin_attribute --img_retrieval --version 3-m
# python3 diffusion_itm.py --task vismin_object --version 3-m
# python3 diffusion_itm.py --task vismin_object --img_retrieval --version 3-m
# python3 diffusion_itm.py --task vismin_counting --version 3-m
# python3 diffusion_itm.py --task vismin_counting --img_retrieval --version 3-m
# python3 diffusion_itm.py --task vlcheck_Relation_vg_action --version 3-m --encoder_drop
# python3 diffusion_itm.py --task vlcheck_Relation_vg_spatial --version 3-m --encoder_drop
# python3 diffusion_itm.py --task vlcheck_Relation_hake --version 3-m --encoder_drop
python3 diffusion_itm.py --task vismin_relation --version 2.0
python3 diffusion_itm.py --task vismin_relation --version 2.0 --img_retrieval
python3 diffusion_itm.py --task vismin_relation --version compdiff --comp_subset spatial
python3 diffusion_itm.py --task vismin_relation --version compdiff --comp_subset non_spatial
python3 diffusion_itm.py --task vismin_relation --version compdiff --comp_subset spatial --img_retrieval
python3 diffusion_itm.py --task vismin_relation --version compdiff --comp_subset non_spatial --img_retrieval
python3 diffusion_itm.py --task vismin_attribute --version 2.0 --img_retrieval
python3 diffusion_itm.py --task vismin_attribute --version 2.0




