#include "Battle.hpp"

Battle::Battle(map<string, map<string, float>> &config) : config(config), BaseScreen("BackGround.png", "Loonboon.ogg")
{
    sun = new Sun(config["Sun"], background_sp.getGlobalBounds());
    attack_interval = 0;
    chosen_card = nullptr;
    deck.push_back(new Card(config["PeaShooter"], background_sp.getGlobalBounds(), "PeaShooter.png"));
    deck.push_back(new Card(config["SnowPea"], background_sp.getGlobalBounds(), "SnowPea.png"));
    deck.push_back(new Card(config["Sunflower"], background_sp.getGlobalBounds(), "Sunflower.png"));
    deck.push_back(new Card(config["Walnut"], background_sp.getGlobalBounds(), "Walnut.png"));
    // deck.push_back(new Card(config["MelonPult"], background_sp.getGlobalBounds(), "MelonPult.png"));
}

State Battle::render(RenderWindow &window)
{
    if (state != BATTLE)
    {
        music.play();
        music.setLoop(true);
        attack_clk.restart();
        state = BATTLE;
    }
    update();
    window.draw(background_sp);
    for_each(zombies.begin(), zombies.end(), [&window](BaseZombie *zombie)
             { zombie->render(window); });
    for_each(deck.begin(), deck.end(), [&window](Card *card)
             { card->render(window); });
    for_each(plants.begin(), plants.end(), [&window](Plant *plant)
             { plant->render(window); });

    sun->render(window);
    window.display();
    return state;
}

void Battle::mouse_press(int x, int y)
{
    if (!sun->mouse_press(x, y))
    {
        auto it = find_if(deck.begin(), deck.end(), [x, y](Card *card)
                          { return card->contains(x, y); });
        if (it != deck.end() && (*it)->ready())
            chosen_card = *it;
        else
            chosen_card = nullptr;
    }
}

void Battle::mouse_release(int x, int y)
{
    if (chosen_card == nullptr)
        return;
    if (!in_battle_feild(x, y) || !sun->can_buy(chosen_card->get_price()))
    {
        chosen_card = nullptr;
        return;
    }
    // this->is_occupied()

    Vector2f pos = find_position(x, y);
    if (!any_of(plants.begin(), plants.end(), [pos](Plant *plant)
                { return plant->get_position() == pos; }))
    {
        cout << "empty" << endl;
        plants.push_back(chosen_card->make_plant(pos));
        cout << plants.size() << endl;
        sun->modify_budget(-chosen_card->get_price());
    }
    chosen_card = nullptr;
}
Vector2f Battle::find_position(int x, int y)
{
    Vector2f nearest_point;
    double min_distance = numeric_limits<double>::max();
    for (unsigned int i = 0; i < WIDTH_GRIDS.size(); ++i)
    {
        for (unsigned int j = 0; j < HEIGHT_GRIDS.size(); ++j)
        {
            Vector2f current_point(background_sp.getGlobalBounds().left + WIDTH_GRIDS[i], background_sp.getGlobalBounds().top + HEIGHT_GRIDS[j]);
            double current_distance = sqrt(pow(current_point.x - x, 2) + pow(current_point.y - y, 2));

            if (current_distance < min_distance)
            {
                min_distance = current_distance;
                nearest_point = current_point;
            }
        }
    }
    return nearest_point;
}

bool Battle::in_battle_feild(int x, int y)
{
    //return BATTLE_FIELD.contains(x - background_sp.getGlobalBounds().left, y - background_sp.getGlobalBounds().top);
    return true;
}

Battle::~Battle()
{
    for_each(zombies.begin(), zombies.end(), [](BaseZombie *zombie)
             { delete zombie; });
    for_each(plants.begin(), plants.end(), [](Plant *plant)
             { delete plant; });
    for_each(deck.begin(), deck.end(), [](Card *card)
             { delete card; });
    delete sun;
}

void Battle::update()
{
    attack();
    find_target();
}

void Battle::find_target()
{
    auto plant_it = plants.begin();
    auto zombie_it = zombies.begin();

    while (plant_it != plants.end())
    {
        if ((*plant_it)->dead())
        {
            plant_it = plants.erase(plant_it);
            continue;
        }
        while (zombie_it != zombies.end())
        {
            if ((*zombie_it)->dead())
            {
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

void Battle::attack()
{
    float elapsed = attack_clk.getElapsedTime().asSeconds();
    if (any_of(zombies.begin(), zombies.end(), [](BaseZombie *zombie)
               { return zombie->win(); }))
    {
        music.stop();
        state = GAMEOVER;
        return;
    }
    else if (elapsed > config["Attacks"]["TotalTime"] && zombies.empty())
    {
        state = VICTORY;
        music.stop();
        return;
    }
    unsigned int num_intervals = config["Attacks"]["TotalTime"] / config["Attacks"]["Interval"];
    for (unsigned int i = 1; i <= num_intervals; i++)
    {
        if ((i - 1) * config["Attacks"]["Interval"] <= elapsed && i * config["Attacks"]["Interval"] > elapsed)
        {
            if (i != attack_interval)
            {
                attack_interval = i;
                make_zombies();
            }
            break;
        }
    }
}

void Battle::make_zombies()
{
    unsigned int num_zombies = config["Attacks"]["StartingNum"] + (attack_interval - 1) * config["Attacks"]["IncNum"];
    unsigned int num_gargantuar = num_zombies * GARGANTUAR_RATIO;
    for (unsigned int i = 0; i < num_gargantuar; i++)
    {
        zombies.push_back(new BaseZombie(config["Gargantuar"], "Gargantuar.png", background_sp.getGlobalBounds()));
    }
    for (unsigned int i = 0; i < (num_zombies - num_gargantuar); i++)
    {
        zombies.push_back(new BaseZombie(config["Zombie"], "Zombie.png", background_sp.getGlobalBounds()));
    }
}