import React from 'react';
import { View, Text, StyleSheet } from 'react-native';

const ChatMessage = ({ message, isUser }) => {
  return (
    <View style={[
      styles.messageContainer,
      isUser ? styles.userMessage : styles.botMessage
    ]}>
      <View style={[
        styles.messageContent,
        isUser ? styles.userMessageContent : styles.botMessageContent
      ]}>
        <Text style={[
          styles.messageText,
          isUser ? styles.userText : styles.botText
        ]}>
          {message}
        </Text>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  messageContainer: {
    marginVertical: 5,
    marginHorizontal: 10,
    maxWidth: '85%',
  },
  userMessage: {
    alignSelf: 'flex-end',
  },
  botMessage: {
    alignSelf: 'flex-start',
  },
  messageContent: {
    borderRadius: 12,
    padding: 12,
  },
  userMessageContent: {
    backgroundColor: '#007AFF',
  },
  botMessageContent: {
    backgroundColor: '#F0F0F0',
  },
  userText: {
    color: '#FFFFFF',
  },
  botText: {
    color: '#000000',
  },
  messageText: {
    fontSize: 16,
    lineHeight: 22,
  },
});

export default ChatMessage;
