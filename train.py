from model import DNNModel

train_datapaths = [
    # '../lake_twolaps',
    # '../jungle_twolaps',
    # '../udacity_data',
    # '../lake_recovery',
    # '../jungle_recovery',
    '../lake_reverse',
    # '../jungle_reverse',
    # '../jungle_corner',
] 

if __name__ == '__main__':
    model = DNNModel()
    model.train(
        train_datapaths,
        epochs=5,
        batch_size=128,
        train_flipped=True
    )
    model.plot()
    model.save('model.h5')
