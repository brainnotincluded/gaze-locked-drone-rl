#!/bin/bash
cd /home/webmaster/gaze-locked-drone-rl
source venv/bin/activate
export PYTHONPATH=/home/webmaster/gaze-locked-drone-rl:$PYTHONPATH
python src/agents/train.py --steps 1000000
