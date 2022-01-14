#generate_data.py
import pickle
import numpy as np 
np.random.seed(0)

def dummy_template():
    """
    Generates a dummy template for a supermarket
    :return S: 2D numpy array of the supermarket.
    """
    S = np.array([
        [1,1,1,1,1,1,1,1,1,1,1],
        [1,0,1,0,1,0,1,0,1,0,1],
        [1,0,1,0,1,0,0,0,1,0,1],
        [1,0,1,0,1,0,1,1,1,1,1],
        [1,0,0,0,1,0,1,0,1,0,1],
        [1,1,1,1,1,1,1,0,1,0,1],
        [1,0,0,0,0,0,1,0,1,0,1],
        [1,1,1,1,1,1,1,0,1,1,1]]).astype(bool)
    return S

def space_template(size=None):
    """
    Generates a template for a supermarket.
    2D array where True indicates a valid space, and False as non-valid (e.g., a display stand)
    :param shapsizee: (L,W) - L: length of supermarket, W: width of the supermarket
    :return S: 2D numpy array of the supermarket.
    """
    if size is not None:
        assert isinstance(size, tuple), 'size must be a tuple'
        assert isinstance(size[0], int) and isinstance(size[1], int), 'size must be a tuple of two integers'
        assert size[0] > 0 and size[1] > 0, 'integers of size must be positive'

        S = np.random.randint(low=0, high=2, size=size).astype(bool)
    else:
        S = dummy_template()

    return S

def dummy_next_square(S, x, y, px, py):   
    """
    Generates the probability to travel to the next square in the supermarket 
    :param S: supermarket template
    :param x: current x position
    :param y: current y position
    :param px: previous x position
    :param py: previous y position

    :return pd: a list of distionaries of {'next':(x,y), 'p': p}
    """
    if x==len(S)-1 and y==len(S[0])-1:
        return None 
    pd = []
    if x<len(S)-1 and S[x+1,y]:
        pp = 0.9 if x>px else 0.1
        pd.append({'next': (x+1,y), 'p':pp})
    if y<len(S[0])-1 and S[x,y+1]:         
        pp = 0.9 if y>py else 0.1
        pd.append({'next': (x,y+1), 'p':pp})
    if x>0 and S[x-1,y]:        
        pp = 0.9 if x<px else 0.1   
        pd.append({'next': (x-1,y), 'p':pp})
    if y>0 and S[x,y-1]:        
        pp = 0.9 if y<py else 0.1
        pd.append({'next': (x,y-1), 'p':pp})
    return pd

def generate_dummy_travels(S, N_samples):
    """
    :param S: supermarket template
    :param N: number of samples to generate
 
    :return travels: a list of positions (x,y)'s
    """
    travels =[]
    for k in range(N_samples):
        x,y,px,py = 0,0,0,0
        pd = dummy_next_square(S, x, y, px, py)
        pos, i = [], 0
        pos.append([x,y])
        while pd is not None:
            proba = [pds['p'] for pds in pd]
            next = np.random.choice(np.arange(len(pd)), 1, p=proba/np.sum(proba))
            px, py = x, y
            x, y = pd[int(next)]['next']
            pos.append([x,y])
            pd = dummy_next_square(S, x, y, px, py)
            i+=1
        travels.append(pos)
    return travels


directions = {'stay':0, 'left':1, 'up':2, 'right':3, 'down':4}

def get_dummy_target(pos_from, pos_to):
    if pos_from[0] == pos_to[0] and pos_from[1] == pos_to[1]:
        target = directions['stay']
    if pos_from[0] >  pos_to[0] and pos_from[1] == pos_to[1]:
        target = directions['down']
    if pos_from[0] <  pos_to[0] and pos_from[1] == pos_to[1]:
        target = directions['up']
    if pos_from[0] == pos_to[0] and pos_from[1] >  pos_to[1]:
        target = directions['left']
    if pos_from[0] == pos_to[0] and pos_from[1] <  pos_to[1]:
        target = directions['right']
    return target

def convert_travels_to_targets(travels):
    target_data = []
    for travel in travels:
        pos_target = []
        for i in range(1,len(travel)):
            target = get_dummy_target(travel[i-1], travel[i])
            pos_target.append(target)
        pos_target.append(0) #end of travel
        target_data += [pos_target]

    return target_data

def generate_dummy_dataset(N_samples=400, path='dataset.pkl', S=None, size=None):
    assert isinstance(path, str), 'path must be a string'
    assert isinstance(N_samples, int) and N_samples>0, 'N_samples must be non-negative integer'

    dataset = {}
    if S is None:
        S = space_template(size)
    travels = generate_dummy_travels(S, N_samples)
    targets = convert_travels_to_targets(travels)

    dataset['travels'] = travels
    dataset['targets'] = targets
    dataset['directions'] = directions

    if path:
        with open(path, "wb") as fp:
            pickle.dump(dataset, fp)  
    return dataset

def load_dataset(path="dataset.pkl"):
    assert isinstance(path, str), 'path must be a string'
    with open(path, "rb") as fp:
        dataset = pickle.load(fp)
    return dataset