# python3 diffusion_itm.py --task winoground --version 3-m
# # python3 diffusion_itm.py --task flickr30k_text  --version 3-m
# python3 diffusion_itm.py --task vg_attribution  --version 3-m
# python3 diffusion_itm.py --task vg_relation --version 3-m
# python3 diffusion_itm.py --task coco_order --version 3-m
# # python3 diffusion_itm.py --task flickr30k_order --version 3-m
# python3 diffusion_itm.py --task clevr --version 3-m
# python3 diffusion_itm.py --task pets --version 3-m
# python3 diffusion_itm.py --task winoground --version 3-m


python3 diffusion_itm.py --task cola_multi --version 2.0
python3 diffusion_itm.py --task cola_multi --version compdiff --comp_subset color
python3 diffusion_itm.py --task cola_multi --version compdiff --comp_subset color --img_retrieval
python3 diffusion_itm.py --task cola_multi --version compdiff --comp_subset shape
python3 diffusion_itm.py --task cola_multi --version compdiff --comp_subset shape --img_retrieval
python3 diffusion_itm.py --task cola_multi --version compdiff --comp_subset texture
python3 diffusion_itm.py --task cola_multi --version compdiff --comp_subset texture --img_retrieval
python3 diffusion_itm.py --task cola_multi --version compdiff --comp_subset complex
python3 diffusion_itm.py --task cola_multi --version compdiff --comp_subset complex --img_retrieval
python3 diffusion_itm.py --task winoground --version 2.0
python3 diffusion_itm.py --task winoground --version 2.0 --img_retrieval
python3 diffusion_itm.py --task winoground --version compdiff --comp_subset color
python3 diffusion_itm.py --task winoground --version compdiff --comp_subset color --img_retrieval
python3 diffusion_itm.py --task winoground --version compdiff --comp_subset shape
python3 diffusion_itm.py --task winoground --version compdiff --comp_subset shape --img_retrieval
python3 diffusion_itm.py --task winoground --version compdiff --comp_subset texture
python3 diffusion_itm.py --task winoground --version compdiff --comp_subset texture --img_retrieval
python3 diffusion_itm.py --task winoground --version compdiff --comp_subset complex
python3 diffusion_itm.py --task winoground --version compdiff --comp_subset complex --img_retrieval

# python3 diffusion_itm.py --task spec_absolute_spatial --batchsize 16
# python3 diffusion_itm.py --task spec_absolute_spatial --img_retrieval --batchsize 16
# python3 diffusion_itm.py --task spec_existence --img_retrieval --batchsize 16
# python3 diffusion_itm.py --task spec_existence --batchsize 16
# python3 diffusion_itm.py --task spec_absolute_size
# python3 diffusion_itm.py --task spec_absolute_size --img_retrieval
# python3 diffusion_itm.py --task spec_spatial
# python3 diffusion_itm.py --task spec_spatial --img_retrieval
# python3 diffusion_itm.py --task spec_count
# python3 diffusion_itm.py --task spec_count --img_retrieval
# python3 diffusion_itm.py --task spec_existence
# python3 diffusion_itm.py --task spec_existence --img_retrieval
# python3 diffusion_itm.py --task spec_relative_size
# python3 diffusion_itm.py --task spec_relative_size --img_retrieval
# python3 diffusion_itm.py --task spec_relative_spatial
# python3 diffusion_itm.py --task spec_relative_spatial --img_retrieval
