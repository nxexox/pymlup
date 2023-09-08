import mlup


up = mlup.UP.load_from_yaml('example-config.yaml', load_model=True)
# load_model=True will calling up.ml.load()
up.web.load()

# If you want to run the application yourself, or add something else to it, use this variable.
# Example with uvicorn: uvicorn mlupapp:app --host 0.0.0.0 --port 80
app = up.web.app

if __name__ == '__main__':
    up.run_web_app()
