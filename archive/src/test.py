from predictor import ScoringService as ss

if __name__ == '__main__':
    print('Model Load:', ss.get_model())
    print(ss.predict(ss.get_inputs(), '2020-12-30'))
    