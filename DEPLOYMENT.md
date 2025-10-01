# Deployment Guide

Guide for deploying Cellex to production environments.

⚠️ **Important**: This is an educational project. For production medical applications, ensure compliance with regulations like HIPAA, GDPR, and obtain necessary certifications.

## Production Considerations

### 1. Security Requirements

#### Backend API
- Use HTTPS with valid SSL certificates
- Implement authentication (JWT, OAuth2)
- Add rate limiting to prevent abuse
- Validate and sanitize all inputs
- Use environment variables for sensitive data
- Enable CORS only for trusted domains
- Implement proper logging and monitoring

#### Data Privacy
- Encrypt data in transit and at rest
- Implement access controls
- Maintain audit logs
- Follow HIPAA guidelines if handling real medical data
- Implement data retention policies
- Ensure secure deletion of sensitive data

### 2. Backend Deployment

#### Using Gunicorn (Recommended)

Install Gunicorn:
```bash
pip install gunicorn
```

Run the API:
```bash
gunicorn -w 4 -b 0.0.0.0:5000 backend.app:app
```

#### Using Docker

Create `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "backend.app:app"]
```

Build and run:
```bash
docker build -t cellex-api .
docker run -p 5000:5000 cellex-api
```

#### Environment Variables

Create `.env` file:
```bash
FLASK_ENV=production
SECRET_KEY=your-secret-key-here
MODEL_PATH=ml_model/models/checkpoints/best_model.pth
MAX_UPLOAD_SIZE=16777216  # 16MB in bytes
ALLOWED_ORIGINS=https://yourdomain.com
```

### 3. Frontend Deployment

#### Static Hosting

The frontend is static HTML/CSS/JS and can be hosted on:
- **Netlify**: Drag and drop the `frontend/` folder
- **Vercel**: Connect your Git repository
- **AWS S3 + CloudFront**: Upload to S3 bucket with static website hosting
- **GitHub Pages**: Enable in repository settings
- **Azure Static Web Apps**: Deploy from GitHub

#### Update API Endpoint

Update `frontend/script.js` to point to your production API:
```javascript
const API_URL = 'https://your-api-domain.com';
```

#### Nginx Configuration

```nginx
server {
    listen 80;
    server_name yourdomain.com;

    root /var/www/cellex/frontend;
    index index.html;

    location / {
        try_files $uri $uri/ /index.html;
    }

    # Proxy API requests
    location /api/ {
        proxy_pass http://localhost:5000/api/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### 4. Model Optimization

#### Model Quantization
Reduce model size and improve inference speed:
```python
import torch
from ml_model.model import create_model

model = create_model()
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# Quantize
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

torch.save(quantized_model.state_dict(), 'quantized_model.pth')
```

#### ONNX Export
For cross-platform deployment:
```python
import torch
from ml_model.model import create_model

model = create_model()
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model, 
    dummy_input, 
    'model.onnx',
    input_names=['input'],
    output_names=['output']
)
```

### 5. Database Integration

For production, store predictions and user data:

```python
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://user:pass@localhost/cellex'
db = SQLAlchemy(app)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(100))
    image_hash = db.Column(db.String(64))
    prediction_class = db.Column(db.String(50))
    confidence = db.Column(db.Float)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
```

### 6. Monitoring and Logging

#### Logging Setup
```python
import logging
from logging.handlers import RotatingFileHandler

handler = RotatingFileHandler('cellex.log', maxBytes=10000000, backupCount=3)
handler.setLevel(logging.INFO)
app.logger.addHandler(handler)
```

#### Monitoring Tools
- **Prometheus + Grafana**: Metrics and dashboards
- **Sentry**: Error tracking
- **New Relic / DataDog**: Application performance monitoring

### 7. CI/CD Pipeline

#### GitHub Actions Example

`.github/workflows/deploy.yml`:
```yaml
name: Deploy Cellex

on:
  push:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    - name: Install dependencies
      run: pip install -r requirements.txt
    - name: Run tests
      run: |
        python -m unittest tests.test_model
        python -m unittest tests.test_dataset

  deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
    - name: Deploy to production
      # Add your deployment steps here
      run: echo "Deploying to production..."
```

### 8. Scaling Strategies

#### Horizontal Scaling
- Use load balancers (e.g., AWS ELB, Nginx)
- Deploy multiple API instances
- Use container orchestration (Kubernetes)

#### Caching
- Redis for caching frequent predictions
- CDN for frontend assets

#### Async Processing
For heavy workloads, use message queues:
```python
from celery import Celery

celery = Celery('cellex', broker='redis://localhost:6379')

@celery.task
def process_image(image_data):
    # Process image asynchronously
    pass
```

### 9. Backup and Recovery

- Regular database backups
- Model versioning
- Disaster recovery plan
- Geographic redundancy

### 10. Compliance Checklist

For medical applications:

- [ ] HIPAA compliance (if in USA)
- [ ] GDPR compliance (if serving EU)
- [ ] FDA approval (if diagnostic tool)
- [ ] CE marking (if selling in Europe)
- [ ] Regular security audits
- [ ] Data encryption at rest and in transit
- [ ] Access control and authentication
- [ ] Audit logging
- [ ] Incident response plan
- [ ] Regular backups and disaster recovery
- [ ] Terms of service and privacy policy
- [ ] User consent management

## Cost Estimation

### AWS Example (Monthly)
- EC2 t3.medium (API): ~$30
- EC2 t3.micro (Frontend): ~$8
- S3 storage (models/data): ~$10
- CloudFront (CDN): ~$20
- RDS (database): ~$25
- Total: ~$93/month

### Cost Optimization
- Use spot instances for non-critical workloads
- Implement auto-scaling
- Use serverless (Lambda, API Gateway) for variable traffic
- Optimize model size
- Use CDN for static assets

## Recommended Architecture

```
┌──────────────┐
│   CloudFront │  CDN
│   (Frontend) │
└──────┬───────┘
       │
       │
┌──────▼───────┐
│  Load        │
│  Balancer    │
└──────┬───────┘
       │
       ├───────────┬───────────┐
       │           │           │
┌──────▼───┐ ┌────▼────┐ ┌────▼────┐
│ API      │ │ API     │ │ API     │
│ Instance │ │Instance │ │Instance │
└──────┬───┘ └────┬────┘ └────┬────┘
       │          │           │
       └──────────┴───────────┘
                  │
         ┌────────▼─────────┐
         │   PostgreSQL     │
         │   Database       │
         └──────────────────┘
```

## Useful Resources

- [Flask Deployment Options](https://flask.palletsprojects.com/en/2.0.x/deploying/)
- [PyTorch Model Optimization](https://pytorch.org/docs/stable/quantization.html)
- [HIPAA Compliance Guide](https://www.hhs.gov/hipaa/index.html)
- [Medical Device Regulations](https://www.fda.gov/medical-devices)

---

**Remember**: This is a demonstration project. Production medical applications require extensive testing, validation, and regulatory approval.
