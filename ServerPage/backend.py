from flask import Flask, Response, render_template
from flask_cors import CORS
from flask_restx import Api, Resource, fields
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.dialects.mysql import JSON
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
import datetime
import time
import cv2
from flask import Flask, request, jsonify
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
CORS(app)

# 新增JWT配置
app.config['JWT_SECRET_KEY'] = 'your-secret-key-here'  # 建议使用环境变量存储
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = datetime.timedelta(hours=2)

# 数据库配置已存在，确保以下配置正确
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:Aa123321@1.92.135.70:3306/AbnormalDetection'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['JSON_AS_ASCII'] = False

# 初始化扩展
db = SQLAlchemy(app)
jwt = JWTManager(app)

# 1. 用户模型
class User(db.Model):
    __tablename__ = 'user'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(255), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    authority = db.Column(db.String(255), nullable=False, default='admin')

    # 关系定义
    cameras = db.relationship('Camera', backref='user', lazy=True, cascade='all, delete-orphan')
    faces = db.relationship('Face', backref='user', lazy=True, cascade='all, delete-orphan')

    __table_args__ = (
        db.CheckConstraint("authority IN ('admin', 'other')", name='check_authority'),
        db.Index('user_username_index', 'username'),
    )

    def set_password(self, password):
        # 使用werkzeug的安全函数生成密码哈希
        self.password = generate_password_hash(password)

    def check_password(self, password):
        # 验证密码哈希
        return check_password_hash(self.password, password)

    def to_dict(self):
        return {
            'id': self.id,
            'username': self.username,
            'authority': self.authority  # 注意：不要返回密码
        }

# 2. 摄像头模型
class Camera(db.Model):
    __tablename__ = 'camera'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    userId = db.Column(db.Integer, db.ForeignKey('user.id', onupdate='CASCADE', ondelete='CASCADE'), nullable=False)
    name = db.Column(db.String(255), unique=True, nullable=False)
    place = db.Column(db.String(255), nullable=False)
    type = db.Column(db.String(255), nullable=False)

    # 关系定义
    dangers = db.relationship('Danger', backref='camera', lazy=True, cascade='all, delete-orphan')
    warnings = db.relationship('Warning', backref='camera', lazy=True, cascade='all, delete-orphan')

    __table_args__ = (
        db.CheckConstraint("type IN ('face', 'danger')", name='chk_valid_type'),
    )

    def to_dict(self):
        return {
            'id': self.id,
            'userId': self.userId,
            'name': self.name,
            'place': self.place,
            'type': self.type
        }

# 3. 危险区域模型
class Danger(db.Model):
    __tablename__ = 'danger'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    cameraId = db.Column(db.Integer, db.ForeignKey('camera.id', onupdate='CASCADE', ondelete='CASCADE'), nullable=False)
    x1 = db.Column(db.Integer, nullable=False)
    x2 = db.Column(db.Integer, nullable=False)
    y1 = db.Column(db.Integer, nullable=False)
    y2 = db.Column(db.Integer, nullable=False)

    def to_dict(self):
        return {
            'id': self.id,
            'cameraId': self.cameraId,
            'x1': self.x1,
            'x2': self.x2,
            'y1': self.y1,
            'y2': self.y2
        }

    shape = db.Column(JSON, nullable=False)

    def to_dict(self):
        return {
            'id': self.id,
            'cameraId': self.cameraId,
            'shape': self.shape
        }

# 4. 人脸特征模型
class Face(db.Model):
    __tablename__ = 'face'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(255), nullable=False)
    # x1-x128特征点
    x1 = db.Column(db.Float, nullable=False)
    x2 = db.Column(db.Float, nullable=False)
    x3 = db.Column(db.Float, nullable=False)
    x4 = db.Column(db.Float, nullable=False)
    x5 = db.Column(db.Float, nullable=False)
    x6 = db.Column(db.Float, nullable=False)
    x7 = db.Column(db.Float, nullable=False)
    x8 = db.Column(db.Float, nullable=False)
    x9 = db.Column(db.Float, nullable=False)
    x10 = db.Column(db.Float, nullable=False)
    x11 = db.Column(db.Float, nullable=False)
    x12 = db.Column(db.Float, nullable=False)
    x13 = db.Column(db.Float, nullable=False)
    x14 = db.Column(db.Float, nullable=False)
    x15 = db.Column(db.Float, nullable=False)
    x16 = db.Column(db.Float, nullable=False)
    x17 = db.Column(db.Float, nullable=False)
    x18 = db.Column(db.Float, nullable=False)
    x19 = db.Column(db.Float, nullable=False)
    x20 = db.Column(db.Float, nullable=False)
    x21 = db.Column(db.Float, nullable=False)
    x22 = db.Column(db.Float, nullable=False)
    x23 = db.Column(db.Float, nullable=False)
    x24 = db.Column(db.Float, nullable=False)
    x25 = db.Column(db.Float, nullable=False)
    x26 = db.Column(db.Float, nullable=False)
    x27 = db.Column(db.Float, nullable=False)
    x28 = db.Column(db.Float, nullable=False)
    x29 = db.Column(db.Float, nullable=False)
    x30 = db.Column(db.Float, nullable=False)
    x31 = db.Column(db.Float, nullable=False)
    x32 = db.Column(db.Float, nullable=False)
    x33 = db.Column(db.Float, nullable=False)
    x34 = db.Column(db.Float, nullable=False)
    x35 = db.Column(db.Float, nullable=False)
    x36 = db.Column(db.Float, nullable=False)
    x37 = db.Column(db.Float, nullable=False)
    x38 = db.Column(db.Float, nullable=False)
    x39 = db.Column(db.Float, nullable=False)
    x40 = db.Column(db.Float, nullable=False)
    x41 = db.Column(db.Float, nullable=False)
    x42 = db.Column(db.Float, nullable=False)
    x43 = db.Column(db.Float, nullable=False)
    x44 = db.Column(db.Float, nullable=False)
    x45 = db.Column(db.Float, nullable=False)
    x46 = db.Column(db.Float, nullable=False)
    x47 = db.Column(db.Float, nullable=False)
    x48 = db.Column(db.Float, nullable=False)
    x49 = db.Column(db.Float, nullable=False)
    x50 = db.Column(db.Float, nullable=False)
    x51 = db.Column(db.Float, nullable=False)
    x52 = db.Column(db.Float, nullable=False)
    x53 = db.Column(db.Float, nullable=False)
    x54 = db.Column(db.Float, nullable=False)
    x55 = db.Column(db.Float, nullable=False)
    x56 = db.Column(db.Float, nullable=False)
    x57 = db.Column(db.Float, nullable=False)
    x58 = db.Column(db.Float, nullable=False)
    x59 = db.Column(db.Float, nullable=False)
    x60 = db.Column(db.Float, nullable=False)
    x61 = db.Column(db.Float, nullable=False)
    x62 = db.Column(db.Float, nullable=False)
    x63 = db.Column(db.Float, nullable=False)
    x64 = db.Column(db.Float, nullable=False)
    x65 = db.Column(db.Float, nullable=False)
    x66 = db.Column(db.Float, nullable=False)
    x67 = db.Column(db.Float, nullable=False)
    x68 = db.Column(db.Float, nullable=False)
    x69 = db.Column(db.Float, nullable=False)
    x70 = db.Column(db.Float, nullable=False)
    x71 = db.Column(db.Float, nullable=False)
    x72 = db.Column(db.Float, nullable=False)
    x73 = db.Column(db.Float, nullable=False)
    x74 = db.Column(db.Float, nullable=False)
    x75 = db.Column(db.Float, nullable=False)
    x76 = db.Column(db.Float, nullable=False)
    x77 = db.Column(db.Float, nullable=False)
    x78 = db.Column(db.Float, nullable=False)
    x79 = db.Column(db.Float, nullable=False)
    x80 = db.Column(db.Float, nullable=False)
    x81 = db.Column(db.Float, nullable=False)
    x82 = db.Column(db.Float, nullable=False)
    x83 = db.Column(db.Float, nullable=False)
    x84 = db.Column(db.Float, nullable=False)
    x85 = db.Column(db.Float, nullable=False)
    x86 = db.Column(db.Float, nullable=False)
    x87 = db.Column(db.Float, nullable=False)
    x88 = db.Column(db.Float, nullable=False)
    x89 = db.Column(db.Float, nullable=False)
    x90 = db.Column(db.Float, nullable=False)
    x91 = db.Column(db.Float, nullable=False)
    x92 = db.Column(db.Float, nullable=False)
    x93 = db.Column(db.Float, nullable=False)
    x94 = db.Column(db.Float, nullable=False)
    x95 = db.Column(db.Float, nullable=False)
    x96 = db.Column(db.Float, nullable=False)
    x97 = db.Column(db.Float, nullable=False)
    x98 = db.Column(db.Float, nullable=False)
    x99 = db.Column(db.Float, nullable=False)
    x100 = db.Column(db.Float, nullable=False)
    x101 = db.Column(db.Float, nullable=False)
    x102 = db.Column(db.Float, nullable=False)
    x103 = db.Column(db.Float, nullable=False)
    x104 = db.Column(db.Float, nullable=False)
    x105 = db.Column(db.Float, nullable=False)
    x106 = db.Column(db.Float, nullable=False)
    x107 = db.Column(db.Float, nullable=False)
    x108 = db.Column(db.Float, nullable=False)
    x109 = db.Column(db.Float, nullable=False)
    x110 = db.Column(db.Float, nullable=False)
    x111 = db.Column(db.Float, nullable=False)
    x112 = db.Column(db.Float, nullable=False)
    x113 = db.Column(db.Float, nullable=False)
    x114 = db.Column(db.Float, nullable=False)
    x115 = db.Column(db.Float, nullable=False)
    x116 = db.Column(db.Float, nullable=False)
    x117 = db.Column(db.Float, nullable=False)
    x118 = db.Column(db.Float, nullable=False)
    x119 = db.Column(db.Float, nullable=False)
    x120 = db.Column(db.Float, nullable=False)
    x121 = db.Column(db.Float, nullable=False)
    x122 = db.Column(db.Float, nullable=False)
    x123 = db.Column(db.Float, nullable=False)
    x124 = db.Column(db.Float, nullable=False)
    x125 = db.Column(db.Float, nullable=False)
    x126 = db.Column(db.Float, nullable=False)
    x127 = db.Column(db.Float, nullable=False)
    x128 = db.Column(db.Float, nullable=False)
    userId = db.Column(db.Integer, db.ForeignKey('user.id', onupdate='CASCADE', ondelete='CASCADE'), nullable=False)

    def to_dict(self):
        # 动态生成x1-x128的字典
        features = {f'x{i}': getattr(self, f'x{i}') for i in range(1, 129)}
        return {
            'id': self.id,
            'name': self.name,
            'userId': self.userId,
            **features
        }

# 5. 警告事件模型
class Warning(db.Model):
    __tablename__ = 'warning'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    cameraId = db.Column(db.Integer, db.ForeignKey('camera.id'), nullable=False)
    curTime = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    videoURL = db.Column(db.String(255), nullable=False)
    info = db.Column(db.String(255), nullable=False)
    type = db.Column(db.String(255), nullable=False)

    __table_args__ = (
        db.CheckConstraint("type IN ('stranger', 'cheat', 'helmet', 'dangerous area', 'tumble')", name='chk_valid_event_type'),
    )

    def to_dict(self):
        return {
            'id': self.id,
            'cameraId': self.cameraId,
            'curTime': self.curTime.isoformat() if self.curTime else None,
            'videoURL': self.videoURL,
            'info': self.info,
            'type': self.type
        }

# 初始化Swagger
api = Api(
    app,
    version='1.0',
    title='异常检测系统API',
    description='异常检测系统API文档',
    doc='/swagger/',
    validate=True,
    default_fields_mask=None  # 添加此行禁用字段掩码
)
ns_stream = api.namespace('streams', description='视频流操作')
ns_data = api.namespace('data', description='数据库操作')
ns_auth = api.namespace('auth', description='用户认证操作')  # 新增认证命名空间

# 定义认证相关的数据模型
register_model = api.model('Register', {
    'username': fields.String(required=True, description='用户名'),
    'password': fields.String(required=True, description='密码'),
    'authority': fields.String(enum=['admin', 'other'], default='other', description='用户权限')
})

login_model = api.model('Login', {
    'username': fields.String(required=True, description='用户名'),
    'password': fields.String(required=True, description='密码')
})

@ns_auth.route('/register')
class RegisterResource(Resource):
    @api.doc(description='用户注册')
    @api.expect(register_model)
    @api.response(201, '用户创建成功')
    @api.response(400, '用户名已存在或参数错误')
    def post(self):
        "用户注册接口"
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        authority = data.get('authority', 'other')

        if not username or not password:
            return {'message': '用户名和密码不能为空'}, 400

        if User.query.filter_by(username=username).first():
            return {'message': '用户名已存在'}, 400

        user = User(username=username, authority=authority)
        user.set_password(password)

        db.session.add(user)
        db.session.commit()

        return {'message': '用户创建成功', 'user': user.to_dict()}, 201

@ns_auth.route('/login')
class LoginResource(Resource):
    @api.doc(description='用户登录')
    @api.expect(login_model)
    @api.response(200, '登录成功')
    @api.response(401, '用户名或密码错误')
    def post(self):
        "用户登录接口，成功后返回JWT令牌"
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')

        user = User.query.filter_by(username=username).first()

        if not user or not user.check_password(password):
            return {'message': '用户名或密码错误'}, 401

        access_token = create_access_token(identity=user.id)

        return {
            'message': '登录成功',
            'access_token': access_token,
            'user': user.to_dict()
        }, 200

# 定义数据模型
user_model = api.model('User', {
    'id': fields.Integer(readOnly=True),
    'username': fields.String(required=True),
    'authority': fields.String(enum=['admin', 'other'], default='admin')
})

# 新增：仅用于创建摄像头的模型（不含id）
camera_create_model = api.model('CameraCreate', {
    'userId': fields.Integer(required=True),
    'name': fields.String(required=True),
    'place': fields.String(required=True),
    'type': fields.String(enum=['face', 'danger'], required=True)
})

# 原摄像头模型保持不变（用于查询和响应）
camera_model = api.model('Camera', {
    'id': fields.Integer(readOnly=True),
    'userId': fields.Integer(required=True),
    'name': fields.String(required=True),
    'place': fields.String(required=True),
    'type': fields.String(enum=['face', 'danger'], required=True)
})

# 新增：危险区域创建模型（不含id）
danger_create_model = api.model('DangerCreate', {
    'cameraId': fields.Integer(required=True),
    'x1': fields.Integer(required=True),
    'x2': fields.Integer(required=True),
    'y1': fields.Integer(required=True),
    'y2': fields.Integer(required=True)
})

danger_model = api.model('Danger', {
    'id': fields.Integer(readOnly=True),
    'cameraId': fields.Integer(required=True),
    'x1': fields.Integer(required=True),
    'x2': fields.Integer(required=True),
    'y1': fields.Integer(required=True),
    'y2': fields.Integer(required=True)
})

# 新增：警告事件创建模型（不含id和curTime）
warning_create_model = api.model('WarningCreate', {
    'cameraId': fields.Integer(required=True),
    'videoURL': fields.String(required=True),
    'info': fields.String(required=True),
    'type': fields.String(enum=['stranger', 'cheat', 'helmet', 'dangerous area', 'tumble'], required=True)
})

warning_model = api.model('Warning', {
    'id': fields.Integer(readOnly=True),
    'cameraId': fields.Integer(required=True),
    'curTime': fields.DateTime(readOnly=True),
    'videoURL': fields.String(required=True),
    'info': fields.String(required=True),
    'type': fields.String(enum=['stranger', 'cheat', 'helmet', 'dangerous area', 'tumble'], required=True)
})

# 新增：人脸特征创建模型（不含id）
face_create_model_fields = {
    'name': fields.String(required=True),
    'userId': fields.Integer(required=True)
}
for i in range(1, 129):
    face_create_model_fields[f'x{i}'] = fields.Float(required=True)
face_create_model = api.model('FaceCreate', face_create_model_fields)

face_model_fields = {
    'id': fields.Integer(readOnly=True),
    'name': fields.String(required=True),
    'userId': fields.Integer(required=True)
}
for i in range(1, 129):
    face_model_fields[f'x{i}'] = fields.Float(required=True)
face_model = api.model('Face', face_model_fields)

# 用户查询API - 单个用户
@ns_data.route('/users/<int:id>')
@api.param('id', '用户ID')
@api.response(404, '用户不存在')
class UserResource(Resource):
    @api.doc(description='根据ID查询用户')
    @api.marshal_with(user_model)
    def get(self, id):
        user = User.query.get(id)
        if not user:
            return {'message': 'User not found'}, 404
        return user

# 新增：用户查询API - 所有用户
@ns_data.route('/users')
class UsersResource(Resource):
    @api.doc(description='获取所有用户数据')
    @api.marshal_list_with(user_model)
    def get(self):
        "获取系统中所有用户信息"
        users = User.query.all()
        return users

# 摄像头查询API - 单个摄像头
@ns_data.route('/cameras/<int:id>')
@api.param('id', '摄像头ID')
@api.response(404, '摄像头不存在')
class CameraResource(Resource):
    @api.doc(description='根据ID查询摄像头')
    @api.marshal_with(camera_model)
    def get(self, id):
        "获取特定摄像头信息"
        camera = Camera.query.get_or_404(id)
        return camera

# 新增：摄像头查询API - 所有摄像头
@ns_data.route('/cameras')
class CamerasResource(Resource):
    @api.doc(description='获取所有摄像头数据')
    @api.marshal_list_with(camera_model)
    def get(self):
        "获取系统中所有摄像头信息"
        cameras = Camera.query.all()
        return cameras
    
    @api.doc(description='添加新摄像头')
    @api.expect(camera_create_model)  # 使用新的创建模型
    @api.response(201, '摄像头创建成功')
    @api.response(400, '参数错误或摄像头名称已存在')
    @api.response(404, '用户不存在')
    def post(self):
        "添加新摄像头数据（ID由数据库自增）"
        data = request.get_json()
        
        # 检查必填字段
        required_fields = ['userId', 'name', 'place', 'type']
        for field in required_fields:
            if field not in data:
                return {'message': f'缺少必填字段: {field}'}, 400
        
        # 检查用户是否存在
        user = User.query.get(data['userId'])
        if not user:
            return {'message': '用户不存在'}, 404
        
        # 检查摄像头名称是否已存在
        if Camera.query.filter_by(name=data['name']).first():
            return {'message': '摄像头名称已存在'}, 400
        
        # 创建新摄像头
        new_camera = Camera(
            userId=data['userId'],
            name=data['name'],
            place=data['place'],
            type=data['type']
        )
        
        db.session.add(new_camera)
        db.session.commit()
        
        return {'message': '摄像头创建成功', 'camera': new_camera.to_dict()}, 201
@ns_data.route('/cameras/by_username/<string:username>')
@api.param('username', '用户名')
@api.response(404, '用户不存在')
class CamerasByUsernameResource(Resource):
    @api.doc(description='根据用户名查询摄像头')
    @api.marshal_list_with(camera_model)
    def get(self, username):
        "获取指定用户名下的所有摄像头"
        user = User.query.filter_by(username=username).first_or_404()
        cameras = Camera.query.filter_by(userId=user.id).all()
        return cameras

# 危险区域查询API - 单个危险区域
@ns_data.route('/dangers/<int:id>')
@api.param('id', '危险区域ID')
@api.response(404, '危险区域不存在')
class DangerResource(Resource):
    @api.doc(description='根据ID查询危险区域')
    @api.marshal_with(danger_model)
    def get(self, id):
        "获取特定危险区域信息"
        danger = Danger.query.get_or_404(id)
        return danger

# 新增：危险区域查询API - 所有危险区域
@ns_data.route('/dangers')
class DangersResource(Resource):
    @api.doc(description='获取所有危险区域数据')
    @api.marshal_list_with(danger_model)
    def get(self):
        "获取系统中所有危险区域信息"
        dangers = Danger.query.all()
        return dangers
    
    @api.doc(description='添加新危险区域')
    @api.expect(danger_create_model)
    @api.response(201, '危险区域创建成功')
    @api.response(400, '参数错误')
    @api.response(404, '摄像头不存在')
    def post(self):
        "添加新危险区域数据（ID由数据库自增）"
        data = request.get_json()
        
        # 检查必填字段
        required_fields = ['cameraId', 'x1', 'x2', 'y1', 'y2']
        for field in required_fields:
            if field not in data:
                return {'message': f'缺少必填字段: {field}'}, 400
        
        # 检查摄像头是否存在
        camera = Camera.query.get(data['cameraId'])
        if not camera:
            return {'message': '摄像头不存在'}, 404
        
        # 创建新危险区域
        new_danger = Danger(
            cameraId=data['cameraId'],
            x1=data['x1'],
            x2=data['x2'],
            y1=data['y1'],
            y2=data['y2']
        )
        
        db.session.add(new_danger)
        db.session.commit()
        
        return {'message': '危险区域创建成功', 'danger': new_danger.to_dict()}, 201
@ns_data.route('/dangers/by_camera_name/<string:camera_name>')
@api.param('camera_name', '摄像头名称')
@api.response(404, '摄像头不存在')
class DangersByCameraNameResource(Resource):
    @api.doc(description='根据摄像头名称查询危险区域')
    @api.marshal_list_with(danger_model)
    def get(self, camera_name):
        "获取指定摄像头的所有危险区域"
        camera = Camera.query.filter_by(name=camera_name).first_or_404()
        dangers = Danger.query.filter_by(cameraId=camera.id).all()
        return dangers

# 警告事件查询API - 单个警告事件
@ns_data.route('/warnings/<int:id>')
@api.param('id', '警告事件ID')
@api.response(404, '警告事件不存在')
class WarningResource(Resource):
    @api.doc(description='根据ID查询警告事件')
    @api.marshal_with(warning_model)
    def get(self, id):
        "获取特定警告事件信息"
        warning = Warning.query.get_or_404(id)
        return warning

# 新增：警告事件查询API - 所有警告事件
@ns_data.route('/warnings')
class WarningsResource(Resource):
    @api.doc(description='获取所有警告事件数据')
    @api.marshal_list_with(warning_model)
    def get(self):
        "获取系统中所有警告事件信息"
        warnings = Warning.query.all()
        return warnings
    
    @api.doc(description='添加新警告事件')
    @api.expect(warning_create_model)
    @api.response(201, '警告事件创建成功')
    @api.response(400, '参数错误')
    @api.response(404, '摄像头不存在')
    def post(self):
        "添加新警告事件数据（ID和curTime由系统自动生成）"
        data = request.get_json()
        
        # 检查必填字段
        required_fields = ['cameraId', 'videoURL', 'info', 'type']
        for field in required_fields:
            if field not in data:
                return {'message': f'缺少必填字段: {field}'}, 400
        
        # 检查摄像头是否存在
        camera = Camera.query.get(data['cameraId'])
        if not camera:
            return {'message': '摄像头不存在'}, 404
        
        # 创建新警告事件
        new_warning = Warning(
            cameraId=data['cameraId'],
            videoURL=data['videoURL'],
            info=data['info'],
            type=data['type'],
        )
        
        db.session.add(new_warning)
        db.session.commit()
        
        return {'message': '警告事件创建成功', 'warning': new_warning.to_dict()}, 201
@ns_data.route('/warnings/by_camera_name/<string:camera_name>')
@api.param('camera_name', '摄像头名称')
@api.response(404, '摄像头不存在')
class WarningsByCameraNameResource(Resource):
    @api.doc(description='根据摄像头名称查询警告事件')
    @api.marshal_list_with(warning_model)
    def get(self, camera_name):
        "获取指定摄像头的所有警告事件"
        camera = Camera.query.filter_by(name=camera_name).first_or_404()
        warnings = Warning.query.filter_by(cameraId=camera.id).all()
        return warnings

@ns_data.route('/warnings/by_username/<string:username>')
@api.param('username', '用户名')
@api.response(404, '用户不存在')
class WarningsByUsernameResource(Resource):
    @api.doc(description='根据用户名查询警告事件')
    @api.marshal_list_with(warning_model)
    def get(self, username):
        "获取指定用户名下所有摄像头的警告事件"
        user = User.query.filter_by(username=username).first_or_404()
        # 获取用户所有摄像头的ID列表
        camera_ids = [camera.id for camera in user.cameras]
        # 查询所有关联摄像头的警告事件
        warnings = Warning.query.filter(Warning.cameraId.in_(camera_ids)).all()
        return warnings

@ns_data.route('/faces/<int:id>')
@api.param('id', '人脸特征ID')
@api.response(404, '人脸特征不存在')
class FaceResource(Resource):
    @api.doc(description='根据ID查询人脸特征')
    @api.marshal_with(face_model)
    def get(self, id):
        "获取特定人脸特征信息"
        face = Face.query.get_or_404(id)
        return face
@ns_data.route('/faces')
class FacesResource(Resource):
    @api.doc(description='获取所有人脸特征数据')
    @api.marshal_list_with(face_model)
    def get(self):
        "获取系统中所有人脸特征信息"
        faces = Face.query.all()
        return faces
    
    @api.doc(description='添加新人脸特征')
    @api.expect(face_create_model)
    @api.response(201, '人脸特征创建成功')
    @api.response(400, '参数错误或人脸名称已存在')
    @api.response(404, '用户不存在')
    def post(self):
        "添加新人脸特征数据（ID由数据库自增）"
        data = request.get_json()
        
        # 检查必填字段
        required_fields = ['name', 'userId'] + [f'x{i}' for i in range(1, 129)]
        for field in required_fields:
            if field not in data:
                return {'message': f'缺少必填字段: {field}'}, 400
        
        # 检查用户是否存在
        user = User.query.get(data['userId'])
        if not user:
            return {'message': '用户不存在'}, 404
        
        # 检查人脸名称是否已存在
        if Face.query.filter_by(name=data['name'], userId=data['userId']).first():
            return {'message': '该用户已存在同名人脸特征'}, 400
        
        # 提取128个人脸特征值
        face_features = {f'x{i}': data[f'x{i}'] for i in range(1, 129)}
        
        # 创建新人脸特征
        new_face = Face(
            name=data['name'],
            userId=data['userId'],** face_features
        )
        
        db.session.add(new_face)
        db.session.commit()
        
        return {'message': '人脸特征创建成功', 'face': new_face.to_dict()}, 201
@ns_data.route('/faces/by_username/<string:username>')
@api.param('username', '用户名')
@api.response(404, '用户不存在')
class FacesByUsernameResource(Resource):
    @api.doc(description='根据用户名查询人脸特征')
    @api.marshal_list_with(face_model)
    def get(self, username):
        "获取指定用户名下的所有人脸特征"
        user = User.query.filter_by(username=username).first_or_404()
        faces = Face.query.filter_by(userId=user.id).all()
        return faces

def process_stream(stream_url, timeout=1):
    cap = cv2.VideoCapture(stream_url)
    start_time = time.time()
    while not cap.isOpened():
        if time.time() - start_time > timeout:
            raise RuntimeError(f"无法打开视频流: {stream_url} (超时{timeout}s)")
        time.sleep(0.1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frame_skip = 5
    frame_count = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        if frame_count % frame_skip == 0:
            ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if not ret:
                frame_count += 1
                continue

            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        frame_count += 1

    cap.release()

@app.route('/')
def index():
    # 修改: 前后端分离后不再提供HTML页面
    return {
        'status': 'success',
        'message': 'Video stream backend is running',
        'endpoints': {
            'video_feed': '/video_feed/<stream_id>'
        }
    }, 200

@ns_stream.route('/video_feed/<stream_id>')
@api.doc(
    params={'stream_id': '视频流ID，例如：camera1'},
    responses={
        200: ('成功返回视频流'),
        500: ('服务器错误')
    },
    description='获取实时视频流，返回MJPEG格式的视频流数据'
)
class VideoFeed(Resource):
    def get(self, stream_id):
        """
        获取实时视频流
        ---        
        成功响应：
          - 200: 返回MJPEG视频流，Content-Type为multipart/x-mixed-replace
        错误响应：
          - 500: 视频流打开失败或处理错误
        """
        stream_url = f'rtmp://1.92.135.70:9090/live/{stream_id}'
        try:
            return Response(process_stream(stream_url),
                            mimetype='multipart/x-mixed-replace; boundary=frame')
        except RuntimeError as e:
            app.logger.error(f"视频流处理失败: {e}")
            return {'message': f"视频流打开失败: {e}"}, 500
        except Exception as e:
            app.logger.error(f"未知错误: {e}")
            return {'message': "视频流处理失败"}, 500

if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # 创建所有数据表
    app.run(debug=False, host='0.0.0.0', port=9080)
