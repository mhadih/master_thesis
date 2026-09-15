#include "Battle.hpp"

Battle::Battle(map<string, map<string, int>>& config): 
config(config), BaseScreen("BackGround.png", "Loonboon.ogg")
{
    clock.restart();
    state = BATTLE;
    sun = new Sun(config["Sun"], background_sp.getGlobalBounds());
    interval = 0;
}

void Battle::render(RenderWindow &window){
    update();
    if(music.getStatus() != Music::Playing)
        music.play();
    window.draw(background_sp);
    for_each(zombies.begin(), zombies.end(), [&window](BaseZombie* zombie){ zombie->render(window); });
    sun->render(window);
    window.display();
}

State Battle::mouse_press(int x, int y){
    sun->mouse_press(x, y);
    for_each(zombies.begin(), zombies.end(), [&window](BaseZombie* zombie){ zombie->render(window); });
    return state;
}

Battle::~Battle(){
    for_each(zombies.begin(), zombies.end(), [](BaseZombie* zombie){ delete zombie; });
    for_each(plants.begin(), plants.end(), [](Plant* plant){ delete plant; });
    delete sun;
}

void Battle::update(){
    attack();
    find_target();
}

void Battle::find_target(){
    auto plant_it = plants.begin();
    auto zombie_it = zombies.begin();

    while(plant_it != plants.end()){
        if((*plant_it)->dead()){
            plant_it = plants.erase(plant_it);
            continue;
        }
        while (zombie_it != zombies.end()){
            if((*zombie_it)->dead()){
                zombie_it = zombies.erase(zombie_it);
                continue;
            }   
            (*zombie_it)->set_target(*plant_it);
            (*plant_it)->set_target(*zombie_it);
            ++zombie_it;
        }
        ++plant_it;
    }
}

void Battle::attack(){
    if(any_of(zombies.begin(), zombies.end(),[](BaseZombie& zombie){ zombie.win(); })){
        state = GAMEOVER;
        return;
    }
    else if (clock.getElapsedTime().asSeconds() >= config["Attacks"]["TotalTime"]){
        state = VICTORY;
        return;
    }
    unsigned int num_intervals = config["Attacks"]["TotalTime"]/config["Attacks"]["Interval"];
    float elapsed = clock.getElapsedTime().asSeconds();
    for (unsigned int i = 1; i <= num_intervals; i++){
        if ((i-1)*config["Attacks"]["Interval"] <= elapsed && i*config["Attacks"]["Interval"] > elapsed){
            if(i != interval){
                interval = i;
                make_zombies();
            }
            break;
        }
    }
}

void Battle::make_zombies(){
    unsigned int num_zombies = config["Attacks"]["StartingNum"] + (interval-1) * config["Attacks"]["IncNum"];
    unsigned int num_gargantuar = num_zombies * GARGANTUAR_RATIO;
    for (unsigned int i = 0; i < num_gargantuar; i++){
        zombies.push_back(new BaseZombie(config["Gargantuar"], "Gargantuar.png", background_sp.getGlobalBounds()));
    }
    for (unsigned int i = 0; i < (num_zombies-num_gargantuar); i++){
        zombies.push_back(new BaseZombie(config["Zombie"], "Zombie.png", background_sp.getGlobalBounds()));
    }
}