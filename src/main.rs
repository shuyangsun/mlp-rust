use std::thread;

struct Point {
    x: f64,
    y: f64,
}

fn print_point(point: Point) {
    println!("{{x: {}, y: {}}}", point.x, point.y);
}

fn print_point_ref(point: &Point) {
    println!("{{x: {}, y: {}}}", point.x, point.y);
}

fn reset_x_then_print(point: &mut Point) {
    point.x = 0.0;
    print_point_ref(point)
}

fn reset_y_then_print(point: &mut Point) {
    point.y = 0.0;
    print_point_ref(point)
}

fn main() {
    // 1. Move by Default
    let p1 = Point { x: 2.1, y: 3.2 };
    // let p2 = p1;
    print_point(p1);
    // print_point(p1);

    // 2. Immutable by Default
    let p3 = Point { x: 1.0, y: -1.2 };
    // p3.x = 3.1;

    let mut p4 = Point { x: 1.0, y: -1.2 };
    p4.x = 8.3;

    // 3. Thread Safety
    let handle1 = thread::spawn(move || reset_x_then_print(&mut p4));
    // let handle2 = thread::spawn(move || reset_y_then_print(&mut p4));

    handle1.join().unwrap();
    // handle2.join().unwrap();
}
