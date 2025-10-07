from app import run_app

if __name__ == '__main__':
    timepoints = ['0h', '6h', '18h', '54h']
    base_path = '../../tangerine_mod/arid1a_networks'
    run_app(timepoints, base_path)
