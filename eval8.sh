# python3 diffusion_itm.py --task sugar_add_att --version 3-m
# python3 diffusion_itm.py --task sugar_add_obj --version 3-m
# python3 diffusion_itm.py --task sugar_replace_att --version 3-m
# python3 diffusion_itm.py --task sugar_replace_obj --version 3-m
# python3 diffusion_itm.py --task sugar_replace_rel --version 3-m
# python3 diffusion_itm.py --task sugar_swap_att --version 3-m
# python3 diffusion_itm.py --task sugar_swap_obj --version 3-m

# python3 diffusion_itm.py --task sugar_add_att --version 3-m --encoder_drop
# python3 diffusion_itm.py --task sugar_add_obj --version 3-m --encoder_drop
# python3 diffusion_itm.py --task sugar_replace_att --version 3-m --encoder_drop
# python3 diffusion_itm.py --task sugar_replace_obj --version 3-m --encoder_drop
# python3 diffusion_itm.py --task sugar_replace_rel --version 3-m --encoder_drop
# python3 diffusion_itm.py --task sugar_swap_att --version 3-m --encoder_drop
# python3 diffusion_itm.py --task sugar_swap_obj --version 3-m --encoder_drop

python3 diffusion_itm.py --task winoground --version 3-m --encoder_drop
# python3 diffusion_itm.py --task flickr30k_text  --version 3-m --encoder_drop
python3 diffusion_itm.py --task vg_attribution  --version 3-m --encoder_drop
python3 diffusion_itm.py --task vg_relation --version 3-m --encoder_drop
python3 diffusion_itm.py --task coco_order --version 3-m --encoder_drop
# python3 diffusion_itm.py --task flickr30k_order --version 3-m --encoder_drop
python3 diffusion_itm.py --task clevr --version 3-m --encoder_drop
# python3 diffusion_itm.py --task pets --version 3-m --encoder_drop
python3 diffusion_itm.py --task winoground --version 3-m --encoder_drop
# python3 diffusion_itm.py --task cola_single_clevr  --version 3-m --encoder_drop

