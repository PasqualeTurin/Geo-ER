import argparse
import config
from functions import prepare_dataset
from train import train_GeoER
from model import GeoER

parser = argparse.ArgumentParser(description='GeoER')
parser.add_argument('-s', '--source', type=str, default='osm_yelp', help='Data source (oms_yelp, osm_fsq')
parser.add_argument('-c', '--city', type=str, default='pit', help='City dataset (sin, edi, tor, pit)')


args = parser.parse_args()
if args.city == 'sin':
    city = 'Singapore'
elif args.city == 'edi':
    city = 'Edinburgh'
elif args.city == 'tor':
    city = 'Toronto'
else:
    city = 'Pittsburgh'
    
    
device = config.device

train_path = config.path + args.source + '/' + args.city + '/train.txt'
n_train_path = config.n_path + args.source + '/' + args.city + '/n_train.json'
train_x, train_coord, train_n, train_y = prepare_dataset(train_path, n_train_path, max_seq_len=config.max_seq_len)


valid_path = config.path + args.source + '/' + args.city + '/valid.txt'
n_valid_path = config.n_path + args.source + '/' + args.city + '/n_valid.json'
valid_x, valid_coord, valid_n, valid_y = prepare_dataset(valid_path, n_valid_path, max_seq_len=config.max_seq_len)


test_path = config.path + args.source + '/' + args.city + '/test.txt'
n_test_path = config.n_path + args.source + '/' + args.city + '/n_test.json'
test_x, test_coord, test_n, test_y = prepare_dataset(test_path, n_test_path, max_seq_len=config.max_seq_len)

print('Succesfully loaded',city,'(',args.source,') dataset')
print('Train size:',len(train_x))
print('Valid size:',len(valid_x))
print('Test size:',len(test_x))


model = GeoER(device=device, dropout=config.dropout)
model = model.to(device)

train_GeoER(model, train_x, train_coord, train_n, train_y, valid_x, valid_coord, valid_n, valid_y, test_x, test_coord, test_n, test_y, device, save_path=config.save_path+city.lower(), epochs=config.epochs, batch_size=config.batch_size, lr=config.lr)
