#!/usr/bin/env python3
"""
speech_recognition_node.py - Offline STT using Vosk
Listens to microphone and publishes recognized text to /speech_recognized
"""

import rospy
from std_msgs.msg import String
import sounddevice as sd
import json
import queue

try:
    from vosk import Model, KaldiRecognizer
except ImportError:
    rospy.logerr("Vosk not installed! Run: pip install vosk sounddevice")
    exit(1)


class SpeechRecognitionNode:
    def __init__(self):
        rospy.init_node('speech_recognition_node')

        # Parameters
        self.model_path = rospy.get_param('~model_path', '/path/to/vosk-model-small-en-us-0.15')
        self.sample_rate = rospy.get_param('~sample_rate', 16000)
        self.enabled = False  # Only listen when at customer

        # Publishers
        self.text_pub = rospy.Publisher('/speech_recognized', String, queue_size=1)

        # Subscribers
        rospy.Subscriber('/flag', String, self.flag_cb)

        # Initialize Vosk
        try:
            self.model = Model(self.model_path)
            self.recognizer = KaldiRecognizer(self.model, self.sample_rate)
            rospy.loginfo("Vosk model loaded successfully")
        except Exception as e:
            rospy.logerr(f"Failed to load Vosk model: {e}")
            rospy.logerr("Download from: https://alphacephei.com/vosk/models")
            return

        # Audio queue for processing
        self.audio_queue = queue.Queue()

        rospy.loginfo("Speech Recognition ready (waiting for customer arrival)")
        self.listen_loop()

    def flag_cb(self, msg: String):
        """Enable/disable listening based on robot state"""
        if msg.data == "customer_reached":
            self.enabled = True
            rospy.loginfo("STT enabled - listening to customer")
        elif msg.data == "home_reached":
            self.enabled = False
            rospy.loginfo("STT disabled - robot returning home")

    def audio_callback(self, indata, frames, time, status):
        """Callback for audio stream"""
        if status and status != sd.CallbackFlags.input_overflow:
            rospy.logwarn(f"Audio status: {status}")
        if self.enabled:
            self.audio_queue.put(bytes(indata))

    def listen_loop(self):
        """Continuously listen and recognize speech"""
        with sd.RawInputStream(
            samplerate=self.sample_rate,
            blocksize=4000,
            dtype='int16',
            channels=1,
            callback=self.audio_callback
        ):
            rospy.loginfo("Audio stream started")

            while not rospy.is_shutdown():
                if not self.enabled:
                    # Clear queue when disabled
                    while not self.audio_queue.empty():
                        self.audio_queue.get()
                    rospy.sleep(0.5)
                    continue

                try:
                    data = self.audio_queue.get(timeout=0.5)

                    if self.recognizer.AcceptWaveform(data):
                        result = json.loads(self.recognizer.Result())
                        text = result.get('text', '').strip()

                        if text:
                            rospy.loginfo(f"Recognized: '{text}'")
                            self.text_pub.publish(String(data=text))

                except queue.Empty:
                    pass
                except Exception as e:
                    rospy.logwarn(f"Recognition error: {e}")

    def shutdown(self):
        """Cleanup on node shutdown"""
        rospy.loginfo("Shutting down speech recognition")


if __name__ == '__main__':
    try:
        node = SpeechRecognitionNode()
    except rospy.ROSInterruptException:
        pass
    finally:
        if 'node' in locals():
            node.shutdown()
