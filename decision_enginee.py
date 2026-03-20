def get_recommendation(emotion, intensity):
    
    # LOW intensity
    if intensity <= 3:
        return (emotion, intensity, "relax and maintain mood", "anytime")
    
    # MEDIUM intensity
    elif intensity <= 6:
        if emotion == "sad":
            return (emotion, intensity, "talk to a friend", "evening")
        elif emotion == "angry":
            return (emotion, intensity, "go for a walk", "immediate")
        elif emotion == "happy":
            return (emotion, intensity, "share your happiness", "anytime")
    
    # HIGH intensity
    else:
        if emotion == "sad":
            return (emotion, intensity, "seek support or journal deeply", "night")
        elif emotion == "angry":
            return (emotion, intensity, "do breathing exercises", "immediate")
        elif emotion == "happy":
            return (emotion, intensity, "celebrate fully", "anytime")
    
    return (emotion, intensity, "reflect quietly", "night")