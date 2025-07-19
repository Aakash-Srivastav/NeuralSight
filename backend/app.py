from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
import base64
from datetime import datetime, timedelta
from PIL import Image
import numpy as np
import io
from tensorflow.keras.models import load_model
from flask_jwt_extended import JWTManager

from dotenv import load_dotenv
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY')
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(days=7)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///disease_prediction.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['JWT_TOKEN_LOCATION'] = ['headers']
app.config['JWT_HEADER_NAME'] = 'Authorization'
app.config['JWT_HEADER_TYPE'] = 'Bearer'

# Initialize extensions
db = SQLAlchemy(app)
jwt = JWTManager(app)
CORS(app)

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    predictions = db.relationship('Prediction_table', backref='user', lazy=True)
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Prediction_table(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    image_path = db.Column(db.String(200), nullable=False)
    image_data = db.Column(db.Text)  # Base64 encoded image for database storage
    disease = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    recommendations = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# ML Model Integration
class DiseasePredictor:
    def __init__(self):
        # Load trained CNN model here

        self.model = load_model('models/new_cnn_model.keras')
        pass
    
    def predict(self, image_path):
        """
        Replace this your actual prediction logic
        """
        # In actual implementation:
        # 1. Load and preprocess the image
        # 2. Make prediction using your CNN model
        # 3. Return disease name, confidence, and recommendations
        
        # preprocessing:
        def preprocess_image(image_path):
            IMG_SIZE = (64, 64)
            img = Image.open(image_path).convert('RGB')
            img = img.resize(IMG_SIZE)  # match model input
            img_array = np.array(img).astype('float32')
            img_array = np.expand_dims(img_array, axis=0)  # add batch dim
            return img_array
        
        # Preprocess and predict
        img = preprocess_image(image_path)
        prediction = self.model.predict(img)

        # Display prediction
        predicted_class = np.argmax(prediction, axis=1)[0]

        diseases = ['Central Serous Chorioretinopathy-Color Fundus', 'Diabetic Retinopathy', 'Disc Edema',
               'Glaucoma', 'Healthy', 'Macular Scar', 'Myopia', 'Pterygium', 'Retinal Detachment', 'Retinitis Pigmentosa']
        
        disease = diseases[predicted_class]

        import random
        # disease = random.choice(diseases)
        confidence = random.uniform(0.7, 0.95)
        
        recommendations = {
            'Central Serous Chorioretinopathy-Color Fundus': 'Central Serous Chorioretinopathy-Color Fundus - Consult a doctor immediately for proper treatment.', 
            'Diabetic Retinopathy': 'Diabetic Retinopathy - Consult a doctor immediately for proper treatment.',
            'Disc Edema': 'Disc Edema - Consult a doctor immediately for proper treatment.',
            'Glaucoma': 'Glaucoma - Consult a doctor immediately for proper treatment.',
            'Healthy': 'Healthy - Continue maintaining good health habits.',
            'Macular Scar': 'Macular Scar - Consult a doctor immediately for proper treatment.',
            'Myopia': 'Myopia - Consult a doctor immediately for proper treatment.',
            'Pterygium': 'Pterygium - Self-isolate and seek medical attention.',
            'Retinal Detachment': 'Retinal Detachment - Seek immediate medical attention for proper diagnosis.',
            'Retinitis Pigmentosa': 'Retinitis Pigmentosa - Consult an oncologist for further evaluation.'
        }
        
        return {
            'disease': disease,
            'confidence': confidence,
            'recommendations': recommendations.get(disease, 'Consult a healthcare professional.')
        }
    
# Initialize predictor
predictor = DiseasePredictor()

# Add JWT error handlers
@jwt.expired_token_loader
def expired_token_callback(jwt_header, jwt_payload):
    return jsonify({'error': 'Token has expired'}), 401

@jwt.invalid_token_loader
def invalid_token_callback(error):
    return jsonify({'error': 'Invalid token'}), 401

@jwt.unauthorized_loader
def missing_token_callback(error):
    return jsonify({'error': 'Authorization token is required'}), 401

# Routes
@app.route('/api/signup', methods=['POST'])
def signup():
    try:
        data = request.get_json()
        
        # Validate input
        if not data.get('name') or not data.get('email') or not data.get('password'):
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Check if user already exists
        if User.query.filter_by(email=data['email']).first():
            return jsonify({'error': 'User already exists'}), 400
        
        # Create new user
        user = User(
            name=data['name'],
            email=data['email']
        )
        user.set_password(data['password'])
        
        db.session.add(user)
        db.session.commit()
        
        return jsonify({'message': 'User created successfully'}), 201
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        
        if not data.get('email') or not data.get('password'):
            return jsonify({'error': 'Missing email or password'}), 400
        
        user = User.query.filter_by(email=data['email']).first()
        
        if user and user.check_password(data['password']):
            access_token = create_access_token(identity=user.id)
            print("Issued token:", access_token)
            return jsonify({
                'token': access_token,
                'user': {
                    'id': user.id,
                    'name': user.name,
                    'email': user.email
                }
            }), 200
        
        return jsonify({'error': 'Invalid credentials'}), 401
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        user_id = None
        
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file:
            # Secure filename and save
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
            filename = timestamp + filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Convert image to base64 for database storage
            with open(file_path, 'rb') as img_file:
                img_data = base64.b64encode(img_file.read()).decode('utf-8')
            
            # Make prediction using model
            prediction_result = predictor.predict(file_path)
            
            # Save prediction to database
            prediction = Prediction_table(
                user_id=user_id,
                image_path=file_path,
                image_data=img_data,
                disease=prediction_result['disease'],
                confidence=prediction_result['confidence'],
                recommendations=prediction_result['recommendations']
            )
            
            db.session.add(prediction)
            db.session.commit()
            
            return jsonify({
                'prediction': prediction_result,
                'message': 'Prediction completed successfully'
            }), 200
            
    except Exception as e:
        print("Exception:", str(e))  # Print full exception
        return jsonify({'error': str(e)}), 500

@app.route('/api/history', methods=['GET'])
# @jwt_required()
def get_history():
    try:
        user_id = None
        
        predictions = Prediction_table.query.filter_by(user_id=user_id)\
                                    .order_by(Prediction_table.timestamp.desc())\
                                    .limit(10).all()
        
        history = []
        for pred in predictions:
            history.append({
                'id': pred.id,
                'disease': pred.disease,
                'confidence': pred.confidence,
                'recommendations': pred.recommendations,
                'timestamp': pred.timestamp.isoformat()
            })
        
        return jsonify({'history': history}), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/image/<int:prediction_id>', methods=['GET'])
def get_image(prediction_id):
    try:
        user_id = None
        
        prediction = Prediction_table.query.filter_by(id=prediction_id, user_id=user_id).first()
        if not prediction:
            return jsonify({'error': 'Prediction not found'}), 404
        
        return jsonify({
            'image_data': prediction.image_data,
            'disease': prediction.disease,
            'confidence': prediction.confidence
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'timestamp': datetime.utcnow().isoformat()}), 200

# # Initialize database
# @app.before_first_request
# def create_tables():
#     db.create_all()

# if __name__ == '__main__':
#     with app.app_context():
#         db.create

if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # <- move this here
    app.run(debug=True)