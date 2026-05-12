#pragma once

#include <godot_cpp/classes/control.hpp>
#include <godot_cpp/classes/texture_rect.hpp>
#include <godot_cpp/classes/texture2d.hpp>

#include <godot_cpp/core/gdvirtual.gen.inc>

using namespace godot;

enum TrafficLightType {
    Traffic_Light_GO,
    Traffic_Light_STOP,
    Traffic_Light_CAUTION,
};

VARIANT_ENUM_CAST(TrafficLightType);

class TrafficLight : public Control {
    GDCLASS(TrafficLight, Control);

    TextureRect *texture_rect;

    Ref<Texture2D> go_texture;
    Ref<Texture2D> stop_texture;
    Ref<Texture2D> caution_texture;

    TrafficLightType light_type;

protected:
    static void _bind_methods();

    void _notification(int p_what); //instead of _ready(), ossible conflict of _ready-function(?)

public:
    void set_go_texture(const Ref<Texture2D> &p_texture);
    Ref<Texture2D> get_go_texture() const;

    void set_stop_texture(const Ref<Texture2D> &p_texture);
    Ref<Texture2D> get_stop_texture() const;

    void set_caution_texture(const Ref<Texture2D> &p_texture);
    Ref<Texture2D> get_caution_texture() const;

    void set_light_type(TrafficLightType p_type);
    TrafficLightType get_light_type() const;

    virtual void show_next_light();

    GDVIRTUAL1RC(TrafficLightType, _get_next_light, TrafficLightType);
    
    TrafficLight();
};

