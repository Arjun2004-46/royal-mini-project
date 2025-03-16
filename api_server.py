from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS
import os
import json
import logging
import uuid
import base64
import time
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Server configuration
API_PORT = 5002

# Notification file path
NOTIFICATIONS_FILE = 'incidents/notifications.json'

def ensure_notifications_file():
    """Ensure notifications file exists and is properly initialized"""
    os.makedirs('incidents', exist_ok=True)
    if not os.path.exists(NOTIFICATIONS_FILE):
        with open(NOTIFICATIONS_FILE, 'w') as f:
            json.dump([], f)

def load_notifications():
    """Load notifications from file"""
    ensure_notifications_file()
    try:
        with open(NOTIFICATIONS_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading notifications: {str(e)}")
        return []

def save_notifications(notifications):
    """Save notifications to file"""
    try:
        with open(NOTIFICATIONS_FILE, 'w') as f:
            json.dump(notifications, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving notifications: {str(e)}")

def add_notification(notification_type, data):
    """Add a new notification"""
    notifications = load_notifications()
    notification = {
        'id': str(uuid.uuid4()),
        'type': notification_type,
        'data': data,
        'timestamp': int(time.time() * 1000),
        'acknowledged': False,
        'ack_time': None
    }
    notifications.append(notification)
    save_notifications(notifications)
    return notification

@app.route('/api/notifications/pending')
def get_pending_notifications():
    """Get all unacknowledged notifications"""
    try:
        notifications = load_notifications()
        pending = [n for n in notifications if not n['acknowledged']]
        return jsonify(pending)
    except Exception as e:
        logger.error(f"Error getting pending notifications: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/notifications/<notification_id>/acknowledge', methods=['POST'])
def acknowledge_notification(notification_id):
    """Mark a notification as acknowledged"""
    try:
        notifications = load_notifications()
        for notification in notifications:
            if notification['id'] == notification_id and not notification['acknowledged']:
                notification['acknowledged'] = True
                notification['ack_time'] = int(time.time() * 1000)
                save_notifications(notifications)
                return jsonify({'status': 'success'})
        return jsonify({'error': 'Notification not found or already acknowledged'}), 404
    except Exception as e:
        logger.error(f"Error acknowledging notification: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/notifications/history')
def get_notification_history():
    """Get notification history with optional filters"""
    try:
        notifications = load_notifications()
        
        # Get query parameters
        type_filter = request.args.get('type')
        acknowledged = request.args.get('acknowledged')
        
        if type_filter:
            notifications = [n for n in notifications if n['type'] == type_filter]
        if acknowledged is not None:
            is_ack = acknowledged.lower() == 'true'
            notifications = [n for n in notifications if n['acknowledged'] == is_ack]
            
        return jsonify(notifications)
    except Exception as e:
        logger.error(f"Error getting notification history: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/notifications', methods=['POST'])
def create_notification():
    """Create a new notification"""
    try:
        data = request.json
        if not data or 'type' not in data or 'data' not in data:
            return jsonify({'error': 'Missing required fields'}), 400
            
        notification = add_notification(data['type'], data['data'])
        return jsonify(notification)
    except Exception as e:
        logger.error(f"Error creating notification: {str(e)}")
        return jsonify({'error': str(e)}), 500

def load_incidents():
    """Load incidents from file with UUID if not present"""
    try:
        incidents_file = os.path.join('incidents', 'incident_log.json')
        if os.path.exists(incidents_file):
            with open(incidents_file, 'r') as f:
                incidents = json.load(f)
                # Add UUID to existing incidents if they don't have one
                modified = False
                for incident in incidents:
                    if 'uuid' not in incident:
                        incident['uuid'] = str(uuid.uuid4())
                        modified = True
                if modified:
                    with open(incidents_file, 'w') as f:
                        json.dump(incidents, f, indent=2)
                return incidents
        return []
    except Exception as e:
        logger.error(f"Error loading incidents: {str(e)}")
        return []

@app.route('/api/incidents')
def get_incidents():
    """Get list of incidents"""
    try:
        incidents = load_incidents()
        
        # Return a simplified list for better performance
        simplified_incidents = []
        for incident in incidents:
            simplified_incidents.append({
                'uuid': incident.get('uuid', ''),
                'type': incident.get('type', ''),
                'timestamp': incident.get('timestamp', ''),
                'confidence': incident.get('confidence', 0)
            })
        
        return jsonify(simplified_incidents)
    except Exception as e:
        logger.error(f"Error getting incidents list: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/incidents/<incident_uuid>')
def get_incident_detail(incident_uuid):
    """Get detailed information about a specific incident"""
    try:
        logger.info(f"Fetching incident detail for UUID: {incident_uuid}")
        incidents = load_incidents()
        logger.info(f"Loaded {len(incidents)} incidents from file")
        
        if len(incidents) > 0:
            logger.info(f"First few UUIDs in the list: {[inc.get('uuid', 'NO_UUID') for inc in incidents[:3]]}")
        
        incident = next((inc for inc in incidents if inc.get('uuid') == incident_uuid), None)
        
        if not incident:
            logger.error(f"Incident not found with UUID: {incident_uuid}")
            return jsonify({'error': 'Incident not found'}), 404
            
        logger.info(f"Found incident: {incident.get('type')} at {incident.get('timestamp')}")
        
        # Add the image data to the incident details
        image_path = os.path.join('incidents', incident['image'])
        if os.path.exists(image_path):
            with open(image_path, 'rb') as img_file:
                image_data = base64.b64encode(img_file.read()).decode('utf-8')
                incident['image_data'] = f"data:image/jpeg;base64,{image_data}"
                logger.info(f"Added image data from {image_path}")
        else:
            logger.warning(f"Image file not found: {image_path}")
        
        return jsonify(incident)
    except Exception as e:
        logger.error(f"Error getting incident detail: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/incidents/<incident_uuid>/image')
def get_incident_image_by_uuid(incident_uuid):
    """Get incident image by UUID"""
    try:
        incidents = load_incidents()
        incident = next((inc for inc in incidents if inc.get('uuid') == incident_uuid), None)
        
        if not incident:
            return jsonify({'error': 'Incident not found'}), 404
            
        image_path = os.path.join('incidents', incident['image'])
        if not os.path.exists(image_path):
            return jsonify({'error': 'Image file not found'}), 404
            
        # Get the requested format from query parameter (default to JPEG)
        image_format = request.args.get('format', 'jpeg').lower()
        
        if image_format == 'base64':
            # Return base64 encoded image
            with open(image_path, 'rb') as img_file:
                image_data = base64.b64encode(img_file.read()).decode('utf-8')
                return jsonify({
                    'image_data': f"data:image/jpeg;base64,{image_data}",
                    'filename': incident['image']
                })
        else:
            # Return the actual image file
            return send_from_directory('incidents', incident['image'], 
                                    mimetype='image/jpeg',
                                    as_attachment=request.args.get('download', 'false').lower() == 'true')
            
    except Exception as e:
        logger.error(f"Error getting incident image: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/incidents/files/<path:filename>')
def get_incident_image(filename):
    """Serve incident images by filename (legacy endpoint)"""
    try:
        if not os.path.exists(os.path.join('incidents', filename)):
            return jsonify({'error': 'Image file not found'}), 404
            
        return send_from_directory('incidents', filename, 
                                 mimetype='image/jpeg',
                                 as_attachment=request.args.get('download', 'false').lower() == 'true')
    except Exception as e:
        logger.error(f"Error serving incident image: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=API_PORT, debug=True)
    except KeyboardInterrupt:
        print("\nShutting down the API server...") 