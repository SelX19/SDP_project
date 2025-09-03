import React, { useState } from "react";
import {
    View,
    Text,
    Switch,
    TextInput,
    TouchableOpacity,
    ScrollView,
    StyleSheet,
    Alert,
} from "react-native";
import { useTheme } from "../context/ThemeContext";

export default function SettingsScreen() {
    const { darkMode, toggleDarkMode } = useTheme();
    const [apiKey, setApiKey] = useState("");

    const handleSaveApiKey = () => {
        if (!apiKey) {
            Alert.alert("Error", "Please enter a valid API key.");
            return;
        }
        Alert.alert("Saved", "Your API key has been saved.");
        // TODO: persist securely using AsyncStorage or expo-secure-store
    };

    const handleGoogleLogin = () => {
        Alert.alert("Google Login", "Google login not implemented yet.");
        // TODO: integrate Google Sign-In API
    };

    return (
        <ScrollView
            contentContainerStyle={[
                styles.container,
                { backgroundColor: darkMode ? "#121212" : "#fff" },
            ]}
        >
            {/* Appearance Section */}
            <Text style={[styles.sectionTitle, { color: darkMode ? "#fff" : "#000" }]}>
                Appearance
            </Text>
            <View style={styles.optionRow}>
                <Text style={[styles.optionLabel, { color: darkMode ? "#fff" : "#000" }]}>
                    Dark Mode
                </Text>
                <Switch value={darkMode} onValueChange={toggleDarkMode} />
            </View>

            {/* Account Section */}
            <Text style={[styles.sectionTitle, { color: darkMode ? "#fff" : "#000" }]}>
                Account
            </Text>
            <View style={styles.optionRow}>
                <Text style={[styles.optionLabel, { color: darkMode ? "#fff" : "#000" }]}>
                    API Key
                </Text>
            </View>
            <TextInput
                placeholder="Enter API key"
                placeholderTextColor="#888"
                value={apiKey}
                onChangeText={setApiKey}
                style={[
                    styles.input,
                    {
                        borderColor: darkMode ? "#555" : "#ccc",
                        color: darkMode ? "#fff" : "#000",
                    },
                ]}
            />
            <TouchableOpacity style={styles.button} onPress={handleSaveApiKey}>
                <Text style={styles.buttonText}>Save API Key</Text>
            </TouchableOpacity>

            <TouchableOpacity style={[styles.button, styles.googleButton]} onPress={handleGoogleLogin}>
                <Text style={styles.buttonText}>Login with Google</Text>
            </TouchableOpacity>
        </ScrollView>
    );
}

const styles = StyleSheet.create({
    container: {
        flexGrow: 1,
        padding: 20,
    },
    sectionTitle: {
        fontSize: 18,
        fontWeight: "bold",
        marginVertical: 10,
    },
    optionRow: {
        flexDirection: "row",
        alignItems: "center",
        justifyContent: "space-between",
        marginBottom: 15,
    },
    optionLabel: {
        fontSize: 16,
    },
    input: {
        borderWidth: 1,
        borderRadius: 8,
        padding: 10,
        marginBottom: 10,
    },
    button: {
        backgroundColor: "#ff0000c9",
        padding: 12,
        borderRadius: 8,
        marginBottom: 15,
        alignItems: "center",
    },
    googleButton: {
        backgroundColor: "#007BFF",
        padding: 12,
        borderRadius: 8,
        marginBottom: 15,
        alignItems: "center",
    },
    buttonText: {
        color: "#fff",
        fontWeight: "bold",
    },
});



