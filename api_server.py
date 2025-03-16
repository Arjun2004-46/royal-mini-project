from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS
import os
import json
import logging
import uuid
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Server configuration
API_PORT = 5002

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