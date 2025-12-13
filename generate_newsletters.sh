#!/bin/bash
# generate_newsletters.sh
# Daily newsletter generation script
# Runs at 8 AM every day

# Log file location
LOG_FILE="/home/ubuntu/newsletter_generation.log"

# Function to log messages
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Start
log_message "========= STARTING NEWSLETTER GENERATION ========="

# Base URL of your FastAPI app (running on same EC2)
BASE_URL="http://localhost:8080"

# List of topics to generate
TOPICS=("technology" "sports" "india_power_projects" "hiring_jobs")

# Generate each newsletter
for TOPIC in "${TOPICS[@]}"
do
    log_message "Generating newsletter for topic: $TOPIC"
    
    # Call your API endpoint
    RESPONSE=$(curl -s -w "HTTP_STATUS:%{http_code}" -X POST \
        "$BASE_URL/api/newsletter/$TOPIC/generate" \
        --max-time 300)  # 5 minute timeout
    
    # Extract HTTP status
    HTTP_STATUS=$(echo "$RESPONSE" | grep -o 'HTTP_STATUS:[0-9]*' | cut -d':' -f2)
    RESPONSE_BODY=$(echo "$RESPONSE" | sed -e 's/HTTP_STATUS:[0-9]*$//')
    
    if [ "$HTTP_STATUS" = "200" ]; then
        log_message "✓ Success: $TOPIC newsletter generation started"
        log_message "  Response: $RESPONSE_BODY"
    else
        log_message "✗ Failed: $TOPIC returned HTTP $HTTP_STATUS"
        log_message "  Error: $RESPONSE_BODY"
    fi
    
    # Wait 5 seconds between requests to avoid overload
    sleep 5
done

# Completion
log_message "========= NEWSLETTER GENERATION COMPLETED ========="
log_message ""