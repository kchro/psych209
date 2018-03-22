echo 'running all tests...'
declare -i NUM_EPOCHS=100
declare -i NUM_ITERS=10
declare -i NUM_LAYERS=2
declare -i NUM_DIM=64
# echo 'japanese native model...'
# python main.py --listener=JN --num_epochs=$NUM_EPOCHS --num_iterations=$NUM_ITERS --num_layers=$NUM_LAYERS --num_dim=$NUM_DIM
# echo 'english native model...'
# python main.py --listener=EN --num_epochs=$NUM_EPOCHS --num_iterations=$NUM_ITERS --num_layers=1 --num_dim=32
# python main.py --listener=EN --num_epochs=$NUM_EPOCHS --num_iterations=$NUM_ITERS --num_layers=1 --num_dim=64
# python main.py --listener=EN --num_epochs=$NUM_EPOCHS --num_iterations=$NUM_ITERS --num_layers=2 --num_dim=32
# python main.py --listener=EN --num_epochs=$NUM_EPOCHS --num_iterations=$NUM_ITERS --num_layers=2 --num_dim=64
echo 'japanese learner model...'
# python main1.py --listener=JL --num_epochs=100 --num_iterations=10 --num_layers=1 --num_dim=32
# echo 'bilingual model...'
# python main.py --listener=BL --num_epochs=$NUM_EPOCHS --num_iterations=$NUM_ITERS --num_layers=1 --num_dim=32
# echo 'bilingual model...'
# python main.py --listener=BL --num_epochs=$NUM_EPOCHS --num_iterations=$NUM_ITERS --num_layers=1 --num_dim=64
noti
