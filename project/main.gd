extends Node2D


func _ready() -> void:
	var summator = Summator.new()
	summator.add(3)
	summator.add(4)
	print(summator.get_total())
	
func _process(_delta: float) -> void:
	pass


func _on_button_pressed() -> void:
	$TrafficLight.show_next_light()


func _on_traffic_light_light_changed(new_light: int) -> void:
	print(new_light)
