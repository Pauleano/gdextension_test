extends TrafficLight

func _get_next_light(p_current_light: TrafficLightType) -> TrafficLightType:
	if p_current_light== TrafficLightType.Traffic_Light_STOP:
		return TrafficLightType.Traffic_Light_GO
	if p_current_light == TrafficLightType.Traffic_Light_GO:
		return TrafficLightType.Traffic_Light_CAUTION
	if p_current_light ==TrafficLightType.Traffic_Light_CAUTION:
		return TrafficLightType.Traffic_Light_STOP
	return TrafficLightType.Traffic_Light_STOP
