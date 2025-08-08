import { HeaderButton } from "react-navigation-header-buttons"
import Ionicons from '@expo/vector-icons/Ionicons';
import colors from "../constants/colors";

export default CustomHeaderButton = (props) => {
    return <HeaderButton
        {...props}
        IconComponent={Ionicons}
        iconSize={26}
        color={props.color ?? colors.primary}

    />
}