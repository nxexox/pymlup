import mlup
from model import MyModel

up = mlup.UP(ml_model=MyModel())
up.ml.load()

if __name__ == "__main__":
    up.run_web_app()
