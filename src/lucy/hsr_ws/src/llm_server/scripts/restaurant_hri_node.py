#!/usr/bin/env python3

import os
import json
import rospy

from std_msgs.msg import String, Bool
from std_srvs.srv import Trigger, TriggerRequest

try:
    from llm_server.srv import ExtractOrder
    LLM_SERVICE_AVAILABLE = True
except ImportError:
    LLM_SERVICE_AVAILABLE = False
    rospy.logwarn('[RestaurantHRI] llm_server.srv.ExtractOrder not available')


_YES_WORDS = {
    "yes", "yeah", "yep", "yup", "correct", "right", "sure",
    "absolutely", "exactly", "affirmative", "confirm", "confirmed",
    "that's right", "that is right", "sounds good", "ok", "okay"
}
_NO_WORDS = {
    "no", "nope", "nah", "wrong", "incorrect", "not right",
    "that's wrong", "that is wrong", "neither", "negative", "restart", "start over"
}


class RestaurantHRINode():

    def __init__(self,
                 guest_number=1,
                 json_dir='/tmp/hri_orders',
                 timeout=30.0,
                 retries=3):

        self.guest_number = guest_number
        self.json_dir = json_dir
        self.timeout = timeout
        self.retries = retries
        self.retry_count = 0

        rospy.init_node('restaurant_hri_node', anonymous=False)
        os.makedirs(self.json_dir, exist_ok=True)

        # Publishers
        self.flag_pub = rospy.Publisher('/flag_in',          String, queue_size=10)
        self.say_pub = rospy.Publisher('/say',              String, queue_size=10)
        self.mic_control_pub = rospy.Publisher('/condition_record', Bool,   queue_size=10)

        # STT service
        rospy.loginfo('[RestaurantHRI] Waiting for speech_recognize service...')
        try:
            rospy.wait_for_service('speech_recognize', timeout=10.0)
            self.stt_service = rospy.ServiceProxy('speech_recognize', Trigger)
            rospy.loginfo('[RestaurantHRI] STT service connected.')
        except rospy.ROSException:
            rospy.logwarn('[RestaurantHRI] STT service not available.')
            self.stt_service = None

        # LLM extraction service
        rospy.loginfo('[RestaurantHRI] Waiting for extractOrder service...')
        if LLM_SERVICE_AVAILABLE:
            try:
                rospy.wait_for_service('extractOrder', timeout=10.0)
                self.llm_service = rospy.ServiceProxy('extractOrder', ExtractOrder)
                rospy.loginfo('[RestaurantHRI] LLM service connected.')
            except rospy.ROSException:
                rospy.logwarn('[RestaurantHRI] LLM service not available.')
                self.llm_service = None
        else:
            self.llm_service = None

        # Subscriber LAST — never fires before init completes
        self.flag_sub = rospy.Subscriber('/flag_out', String, self.flag_callback)
        rospy.loginfo('[RestaurantHRI] Node ready.')

    def _say(self, text: str):
        rospy.loginfo('[RestaurantHRI] SAY: "%s"', text)
        self.say_pub.publish(String(data=text))
        rospy.sleep(max(1.0, len(text.split()) * 0.4))

    def _listen(self) -> str:
        """
        Enable mic, call Whisper STT once, disable mic.
        Returns the transcript string or '' on failure.
        """
        if self.stt_service is None:
            return ''
        try:
            self.mic_control_pub.publish(Bool(data=True))
            resp = self.stt_service(TriggerRequest())
            self.mic_control_pub.publish(Bool(data=False))
            return resp.message.strip() if resp.success else ''
        except rospy.ServiceException as e:
            rospy.logerr('[GetGuestInfo] STT call failed: %s', e)
            self.mic_control_pub.publish(Bool(data=False))
            return ''

    def _keyword_yes_no(self, text: str):
        t = text.lower()
        if any(w in t for w in _YES_WORDS):
            return True
        if any(w in t for w in _NO_WORDS):
            return False
        return None

    def _extract_order(self, raw_text: str):

        try:
            result = self.llm_service(text=raw_text)

            rospy.loginfo('[RestaurantHRI] LLM raw reply: "%s"', result.order_json)

            if not result.order_json or not result.order_json.strip():
                rospy.logwarn('[RestaurantHRI] LLM returned empty order_json.')
                return None

            raw = result.order_json.strip()

            try:
                order_data = json.loads(raw)

            except json.JSONDecodeError:
                rospy.logwarn('[RestaurantHRI] Non-JSON output from LLM. Converting.')
                items = [i.strip() for i in raw.split(",") if i.strip()]
                order_data = {"items": items}

            rospy.loginfo('[RestaurantHRI] Extracted order: %s', order_data)
            return order_data

        except rospy.ServiceException as e:
            rospy.logerr('[RestaurantHRI] LLM service failed: %s', e)
            return None

    def _order_to_speech(self, order_data) -> str:
        """Convert order dict/list to a readable string for TTS."""
        if isinstance(order_data, str):
            return order_data
        if isinstance(order_data, list):
            return ', '.join(str(i) for i in order_data)
        if isinstance(order_data, dict):
            items = order_data.get('items') or order_data.get('order') or list(order_data.values())
            if isinstance(items, list):
                return ', '.join(str(i) for i in items)
            return str(items)
        return str(order_data)

    # =========================================================================
    # Main callback
    #
    # FLOW:
    #   1. LISTEN  — capture raw order via Whisper STT
    #   2. EXTRACT — send raw text to LLM, get structured items
    #   3. CONFIRM — read extracted items back to customer, ask yes/no
    #                  NO/ambiguous → back to step 1
    #                  YES          → continue
    #   4. SAVE + publish "order_taken"
    # =========================================================================

    def _order_to_speech(self, order_data) -> str:
        """Convert order dict/list to a readable string for TTS."""
        if isinstance(order_data, str):
            items = [i.strip() for i in order_data.split(',')]
        elif isinstance(order_data, list):
            items = [str(i) for i in order_data]
        elif isinstance(order_data, dict):
            raw = order_data.get('items') or order_data.get('order') or list(order_data.values())
            items = [str(i) for i in raw] if isinstance(raw, list) else [str(raw)]
        else:
            return str(order_data)

        if len(items) == 1:
            return items[0]
        return ', '.join(items[:-1]) + ' and ' + items[-1]

    def flag_callback(self, msg):
        if msg.data == "customer_reached":

            rospy.loginfo('[RestaurantHRI] Customer reached — starting order flow.')

            if self.stt_service is None or self.llm_service is None:
                rospy.logerr('[RestaurantHRI] Required services not available.')
                self._retry()
                return

            MAX_RESTARTS = 3

            for attempt in range(MAX_RESTARTS):

                # ── STEP 1: Listen for the order
                self._say("Hi, my name is Lucy! What would you like to order today?")
                raw_order = self._listen()

                if not raw_order:
                    rospy.logwarn('[RestaurantHRI] No speech detected.')
                    self._say("I didn't catch that. Could you please repeat your order?")
                    continue

                # ── STEP 2: Extract structured items with LLM
                rospy.loginfo('[RestaurantHRI] Sending to LLM for extraction: "%s"', raw_order)
                order_data = self._extract_order(raw_order)

                if order_data is None:
                    self._say("Sorry, I had trouble understanding that. Could you repeat your order?")
                    continue

                # ── STEP 3: Read back extracted items and confirm
                order_str = self._order_to_speech(order_data)
                self._say(f"I have extracted the following items: {order_str}. Is that correct?")
                confirmation = self._listen()

                if not confirmation:
                    rospy.logwarn('[RestaurantHRI] No confirmation speech detected.')
                    self._say("I didn't hear a response. Let's try again.")
                    continue

                verdict = self._keyword_yes_no(confirmation)
                rospy.loginfo('[RestaurantHRI] Confirmation verdict: %s', verdict)

                if verdict is not True:
                    # NO or ambiguous — restart from step 1
                    self._say("I'm sorry about that! Let me take your order again.")
                    continue

                # ── STEP 4: Save and signal success
                json_path = os.path.join(self.json_dir, f'order{self.guest_number}.json')
                with open(json_path, 'w') as f:
                    json.dump({'order': order_data}, f, indent=4)
                rospy.loginfo('[RestaurantHRI] Order saved to %s', json_path)

                self._say("Perfect! Your order has been placed. Thank you!")
                self.guest_number += 1
                self.retry_count = 0
                self.flag_pub.publish(String(data="order_taken"))
                return  # done — wait for next customer_reached

            rospy.logerr('[RestaurantHRI] Failed to collect order after %d attempts.', MAX_RESTARTS)
            self._retry()

        if msg.data == "bar_reached":
            path = self.json_dir + f"/order{self.guest_number - 1}.json"

            with open(path, "r") as f:
                data = json.load(f)

            if "placed" not in data:
                data["placed"] = True

                order = data["order"]["items"]
                self._say(
                    f"Hello, I would like to place an order for delivery. The order is {order}, Please place the items in my tray and say items placed when you are done")

            else:
                path = self.json_dir + f"/order{self.guest_number}.json"
                with open(path, "r") as f:
                    data = json.load(f)
                data["placed"] = True
                order = data["order"]["items"]
                order_str = self._order_to_speech(order)
                self._say(
                    f"Hello, I would like to place an order for delivery. The order is {order_str}, Please place the items in my tray and say items placed when you are done")

            # if msg.data == "bar_reached":
            rospy.loginfo('[RestaurantHRI] Bar reached — waiting for items to be placed.')

            TIMEOUT_SECS = 10.0
            start_time = rospy.Time.now()
            asked_once = False

            while not rospy.is_shutdown():

                # ── Check if 10 s have passed without a result
                elapsed = (rospy.Time.now() - start_time).to_sec()
                if elapsed >= TIMEOUT_SECS and not asked_once:
                    self._say("Have the items been placed on the tray?")
                    asked_once = True

                # ── Listen for speech
                text = self._listen()

                if not text:
                    rospy.sleep(0.5)
                    continue

                rospy.loginfo('[RestaurantHRI] Heard at bar: "%s"', text)

                # ── If we already asked, treat reply as a yes/no confirmation ─
                if asked_once:
                    verdict = self._keyword_yes_no(text)
                    if verdict is True:
                        rospy.loginfo('[RestaurantHRI] Items confirmed placed.')
                        self._say("Thank you for placing the items.")
                        self.flag_pub.publish(String(data="items_ready"))
                        return
                    elif verdict is False:
                        rospy.loginfo('[RestaurantHRI] Not placed yet — resetting timer.')
                        start_time = rospy.Time.now()   # reset timer
                        asked_once = False
                        continue
                    # ambiguous → keep listening
                    continue

                # ── Normal detection before timeout
                if "items placed" in text.lower():
                    rospy.loginfo('[RestaurantHRI] Detected "items placed".')
                    self._say("Thank you for placing the items.")
                    self.flag_pub.publish(String(data="items_ready"))
                    return
                rospy.loginfo('[RestaurantHRI] Bar reached — ready for delivery.')
                raw_order = self._listen()
                if raw_order:
                    rospy.loginfo('[RestaurantHRI] Heard at bar: "%s"', raw_order)
                    if "items placed" in raw_order.lower():
                        rospy.loginfo('[RestaurantHRI] Detected "items placed" at bar.')
                        self._say("Thank you. For placing the items.")
                        self.flag_pub.publish(String(data="items_ready"))

        if msg.data == "delivery_complete":
            rospy.loginfo('[RestaurantHRI] Delivery complete — ready for next customer.')
            self._say("Thank you for dining with us! here are you items have a great day!")

    def _retry(self):
        if self.retry_count >= self.retries:
            self.retry_count = 0
            rospy.logerr('[RestaurantHRI] Max retries reached.')
            return 'failed_after_retrying'
        self.retry_count += 1
        rospy.logwarn('[RestaurantHRI] Retry %d/%d', self.retry_count, self.retries)
        return 'failed'

    def wait_until_items_placed(self):

        rospy.loginfo("Listening until 'items placed' is detected...")

        while not rospy.is_shutdown():

            transcript = self._listen()

            if not transcript:
                rospy.logwarn("No speech detected, listening again...")
                continue

            rospy.loginfo("Heard: %s", transcript)

            if "items placed" in transcript.lower():
                rospy.loginfo("Detected 'items placed'")
                self._say("Thank you. I detected the items are placed.")
                return True


if __name__ == '__main__':
    try:
        node = RestaurantHRINode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
