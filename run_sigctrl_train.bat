python sumo_train \
-e signal_control\smartcross\envs\sumo_3roads_multi_agent_config.yaml \
-d signal_control\entry\sumo_config\sumo_3roads_mappo_baseline.py 


python sumo_train \
-e smartcross/envs/sumo_3roads_multi_agent_config.yaml \
-d entry/sumo_config/sumo_3roads_mappo_baseline.py 

python3 sumo_train \
-e../smartcross/envs/sumo_3roads_multi_agent_config.yaml \
-d entry/sumo_config/sumo_3roads_mappo_baseline.py

python3 sumo_train \
-e../smartcross/envs/sumo_7roads_multi_agent_config.yaml \
-d entry/sumo_config/sumo_7roads_mappo_sota.py