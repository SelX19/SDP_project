import "react-native-url-polyfill/auto";
import { StatusBar } from 'expo-status-bar';
import { StyleSheet, Text, View } from 'react-native';
import { NavigationContainer } from '@react-navigation/native';
import MainNavigator from './components/MainNavigator';
import { useFonts } from 'expo-font';
import * as SplashScreen from 'expo-splash-screen'; //for loading - splash screen
import { useCallback } from "react";
import { HeaderButtonsProvider } from "react-navigation-header-buttons/HeaderButtonsProvider";
import { SafeAreaProvider } from "react-native-safe-area-context";

SplashScreen.preventAutoHideAsync(); //prevnt splash screen from going away, remove after loading sth

export default function App() {

  const [fontsLoaded, error] = useFonts({
    "regular": require("./assets/fonts/Poppins-Regular.ttf")
    //add others in the same way
  })

  const onLayoutRootView = useCallback(async () => {
    if (fontsLoaded) {
      //hide the splash screen
      await SplashScreen.hideAsync();
    }
  }, [fontsLoaded]) //that what is in [] is a condition - when that - then the function above executes

  if (!fontsLoaded) {
    return null;
  }

  return (
    <SafeAreaProvider>
      <View style={{ flex: 1 }} onLayout={onLayoutRootView}>
        <NavigationContainer>
          <HeaderButtonsProvider stackType="js">
            <MainNavigator />
          </HeaderButtonsProvider>
        </NavigationContainer>
      </View >
    </SafeAreaProvider>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center',
  },
});
