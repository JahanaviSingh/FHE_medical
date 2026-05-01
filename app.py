"""
FHE Medical Image Processor
M.Tech Research Project
"""

from flask import Flask
from flask_cors import CORS
from backend.routes.image_routes import image_bp
from backend.routes.fhe_routes import fhe_bp
from backend.routes.image_routes import validation_bp
import os

def create_app():
    app = Flask(
        __name__,
        template_folder="frontend/templates",
        static_folder="frontend/static"
    )

    # Config
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB upload limit
    app.config["UPLOAD_FOLDER"] = "data/uploads"
    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-key-change-in-production")

    # Create upload folder if missing
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    # Enable CORS (needed when frontend runs separately during dev)
    CORS(app)

    # Register route blueprints
    app.register_blueprint(image_bp,      url_prefix="/api/image")
    app.register_blueprint(fhe_bp,        url_prefix="/api/fhe")
    app.register_blueprint(validation_bp, url_prefix="/api/validate")

    # Serve the main page
    from flask import render_template
    @app.route("/")
    def index():
        return render_template("index.html")

    return app


if __name__ == "__main__":
    app = create_app()
    print("\n=== FHE Medical Image Processor ===")
    print("  Open http://localhost:5000 in your browser\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
