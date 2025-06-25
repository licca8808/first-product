from flask import Flask # type: ignore

def create_app():
    app = Flask(__name__, template_folder='../templates', static_folder='../static')

    
    from .routes import main # type: ignore
    app.register_blueprint(main)
    
    return app
