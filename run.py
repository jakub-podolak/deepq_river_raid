from time import sleep

from environment import Environment
from models.random import RandomModel
from models.deepq import DeepQModel

SELECTED_MODEL = DeepQModel
HYPERPARAMS = {}

if __name__ == "__main__":
    env = Environment()
    model = SELECTED_MODEL(env, HYPERPARAMS)
    env.set_window_name(model.get_name())

    print("***** TRAINING ******")
    model.train()

    print("***** EVALUATING RUNS ******")
    score = 0
    run_number = 0

    env.reset_state()
    next_state = env.get_current_state()
    running = True 
    while running: 
        action = model.evaluate(next_state)
        next_state, reward, collision = env.step(action)
        score += reward 
        env.set_window_name(model.get_name() + f' [{int(run_number)}, {int(score)}]')
        env.render()
        
        if collision:
            sleep(1)
            env.reset_state()
            score = 0
            run_number += 1

        if env.should_close():
            env.close_window()
    
        env.frame_sleep(30)