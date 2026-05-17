#!/usr/bin/env python3
"""
whisper_service_node.py — ROS Speech-to-Text Service using Whisper

This ROS node wraps the python-speech-recognition library to provide an 
on-demand transcription service. It listens to a specific microphone and 
uses the local OpenAI Whisper model to transcribe the spoken audio.
"""

import rospy
from std_srvs.srv import Trigger, TriggerResponse
import speech_recognition as sr


class WhisperServiceNode:
    """
    A ROS node that provides a 'speech_recognize' service. When called,
    it records audio from the microphone and returns the transcribed text.
    """

    def __init__(self):
        """Initializes the ROS node, configures the recognizer, and starts the service."""
        rospy.init_node('whisper_service_node')

        # Initialize the speech recognizer object
        self.recognizer = sr.Recognizer()

        # The following block is commented out to skip automatic noise adjustment.
        # # Adjust for ambient noise on startup
        # with sr.Microphone() as source:
        #     rospy.loginfo("[WhisperService] Adjusting for ambient noise. Please wait 2 seconds...")
        #     self.recognizer.adjust_for_ambient_noise(source, duration=2)

        # HARDCODE THE SENSITIVITY INSTEAD OF GUESSING
        # A lower number makes the microphone more sensitive to quiet voices.
        # 300 is a good baseline for typical indoor environments.
        self.recognizer.energy_threshold = 300
        # Allow the recognizer to slightly adjust its threshold dynamically based on the room
        self.recognizer.dynamic_energy_threshold = True

        # Start the ROS service that the Human-Robot Interaction (HRI) node will call
        self.srv = rospy.Service('speech_recognize', Trigger, self.handle_speech_recognize)
        rospy.loginfo("[WhisperService] Ready! Service 'speech_recognize' is active.")

    def handle_speech_recognize(self, req):
        """
        Callback executed whenever the 'speech_recognize' service is called.

        Args:
            req (TriggerRequest): The incoming service request (empty for a Trigger).

        Returns:
            TriggerResponse: Contains a boolean success flag and the transcribed text 
                             (or an error message) in the 'message' field.
        """
        # SAFETY BUFFER: Give the audio hardware 1 second to breathe before reopening.
        # This prevents "Device or resource busy" errors if called repeatedly.
        rospy.sleep(1.0)

        rospy.loginfo("[WhisperService] MIC OPEN. Listening for speech...")
        try:
            # Open the microphone for recording.
            # Note: device_index=8 specifically targets a known hardware microphone.
            with sr.Microphone(device_index=8) as source:
                # Listen for speech.
                # timeout=10.0: Wait up to 10 seconds for someone to start speaking.
                # phrase_time_limit=15.0: Stop recording after 15 seconds of continuous speech.
                audio = self.recognizer.listen(source, timeout=10.0, phrase_time_limit=15.0)

            rospy.loginfo("[WhisperService] Audio captured! Transcribing with Whisper...")

            # Use the local "base.en" (English) Whisper model to transcribe the audio.
            text = self.recognizer.recognize_whisper(audio, model="base.en").strip()

            rospy.loginfo(f"[WhisperService] Heard: \"{text}\"")

            # SAFETY BUFFER: Give the hardware a moment to fully release the microphone
            # before returning and allowing another service call.
            rospy.sleep(0.5)

            # Return successful transcription back to the caller
            return TriggerResponse(success=True, message=text)

        except sr.WaitTimeoutError:
            # Triggered if 10 seconds pass without anyone starting to speak
            rospy.logwarn("[WhisperService] Listening timed out. Nobody spoke.")
            return TriggerResponse(success=False, message="")

        except sr.UnknownValueError:
            # Triggered if audio was recorded but Whisper couldn't understand any words
            rospy.logwarn("[WhisperService] Audio was garbled. Could not understand.")
            return TriggerResponse(success=False, message="")

        except Exception as e:
            # Catch-all for other errors (e.g., microphone disconnected, model loading failed)
            rospy.logerr(f"[WhisperService] Error: {e}")
            return TriggerResponse(success=False, message=str(e))


if __name__ == '__main__':
    try:
        # Instantiate the node and keep it running
        WhisperServiceNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        # Handle manual termination (Ctrl+C) gracefully
        pass
