#!/usr/bin/env python3
"""
dialog_manager.py - Handles customer conversation at arrival
Listens to /speech_recognized, processes queries, responds via /speak
"""

import rospy
from std_msgs.msg import String


class DialogManager:
    def __init__(self):
        rospy.init_node('dialog_manager')

        self.enabled = False
        self.last_response_time = rospy.Time(0)
        self.cooldown = 3.0  # seconds between responses

        # Publishers
        self.speech_pub = rospy.Publisher('/speak', String, queue_size=5)

        # Subscribers
        rospy.Subscriber('/speech_recognized', String, self.stt_cb)
        rospy.Subscriber('/flag', String, self.flag_cb)

        # Simple keyword-based responses (you can enhance with NLP later)
        self.responses = {
            'menu': "Our menu includes sandwiches, salads, drinks, and desserts. What would you like?",
            'water': "Certainly! I will bring you water right away.",
            'coffee': "Would you like regular coffee or espresso?",
            'help': "I can take your order, answer questions about the menu, or call a staff member.",
            'staff': "I'm calling a staff member for you now.",
            'bill': "Let me get your bill. One moment please.",
            'thank': "You're very welcome! Enjoy your meal!",
            'bye': "Thank you for visiting! Have a great day!",
            'toilet': "The restroom is located near the entrance on your left.",
            'wifi': "The WiFi password is written on the menu card on your table."
        }

        rospy.loginfo("Dialog Manager ready")

    def flag_cb(self, msg: String):
        """Enable/disable dialog based on robot state"""
        if msg.data == "customer_reached":
            self.enabled = True
            rospy.loginfo("Dialog enabled - ready for customer questions")
        elif msg.data == "home_reached":
            self.enabled = False
            rospy.loginfo("Dialog disabled")

    def stt_cb(self, msg: String):
        """Process recognized speech and respond"""
        if not self.enabled:
            return

        # Cooldown to prevent rapid-fire responses
        if (rospy.Time.now() - self.last_response_time).to_sec() < self.cooldown:
            return

        text = msg.data.lower().strip()
        rospy.loginfo(f"Processing: '{text}'")

        # Find matching keyword
        response = self.find_response(text)

        if response:
            self.speak(response)
        else:
            # Fallback for unrecognized queries
            self.speak("I'm sorry, I didn't understand that. Could you please repeat?")

    def find_response(self, text):
        """Match keywords to responses"""
        for keyword, response in self.responses.items():
            if keyword in text:
                return response

        # Check for common question patterns
        if any(word in text for word in ['what', 'where', 'how', 'can you']):
            return "I can help with that. Please ask about the menu, services, or call a staff member."

        return None

    def speak(self, text):
        """Publish to TTS"""
        self.speech_pub.publish(String(data=text))
        self.last_response_time = rospy.Time.now()
        rospy.loginfo(f"[RESPONSE] {text}")


if __name__ == '__main__':
    try:
        node = DialogManager()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
