from app import get_app

if __name__ == '__main__':
    timepoints = ['0h', '6h', '18h', '54h']
    base_path = '../../tangerine_mod/arid1a_networks'
    get_app(timepoints, base_path)
