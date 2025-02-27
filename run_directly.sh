conda activate MPP

run_name="demo"
config="basic_config"   # options are "basic_config" for all or swe_only/comp_only/incomp_only/swe_and_incomp
yaml_config="./config/mpp_avit_ti_config.yaml"

python train_basic.py --run_name $run_name --config $config --yaml_config $yaml_config
