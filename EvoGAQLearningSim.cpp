/************************************************************
 * 進化シミュレーション + 遺伝的アルゴリズム (GA) + 簡易Q学習 (RL)
 * 
 * 「毒耐性付き＋Q学習ロジック改善」バージョン
 ************************************************************/

#include <SFML/Graphics.hpp>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <memory>
#include <algorithm>
#include <string>
#include <iostream>
#include <map>

//----------------------------------------------------------
// ユーティリティ関数
//----------------------------------------------------------
float getRandomFloat(float minVal, float maxVal) {
    float t = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    return minVal + t * (maxVal - minVal);
}

// 距離計算
float distance2(sf::Vector2f a, sf::Vector2f b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    return dx*dx + dy*dy; // 2乗距離を返す
}

//----------------------------------------------------------
// 遺伝子情報 (GA 用)
//----------------------------------------------------------
struct Genes {
    float speed;            // 移動速度
    float attack;           // 攻撃力
    bool poison;            // 毒性の有無
    int   legs;             // 脚の本数
    float senseRange;       // 感知範囲
    float poisonResistance; // 毒耐性(0.0 ~ 1.0程度を想定)

    static Genes crossoverAndMutate(const Genes& g1, const Genes& g2) {
        Genes child;
        // 親どちらかから引き継ぐ (50%の確率)
        child.speed           = (rand() % 2 == 0) ? g1.speed : g2.speed;
        child.attack          = (rand() % 2 == 0) ? g1.attack : g2.attack;
        child.poison          = (rand() % 2 == 0) ? g1.poison : g2.poison;
        child.legs            = (rand() % 2 == 0) ? g1.legs   : g2.legs;
        child.senseRange      = (rand() % 2 == 0) ? g1.senseRange : g2.senseRange;
        child.poisonResistance= (rand() % 2 == 0) ? g1.poisonResistance : g2.poisonResistance;

        // 突然変異 (確率は適宜調整)
        if (rand() % 100 < 10) child.speed += getRandomFloat(-0.5f, 0.5f);
        if (rand() % 100 < 10) child.attack += getRandomFloat(-1.f, 1.f);
        if (rand() % 100 < 10) child.senseRange += getRandomFloat(-20.f, 20.f);
        if (rand() % 100 < 5) {
            child.legs += (rand() % 3) - 1; // -1, 0, +1
            if (child.legs < 1) child.legs = 1;
        }
        if (rand() % 100 < 5) {
            child.poison = !child.poison;
        }
        if (rand() % 100 < 10) {
            // 0.0 ~ 1.0 の範囲で少し変異
            child.poisonResistance += getRandomFloat(-0.2f, 0.2f);
        }

        // 範囲制限
        if (child.speed < 10.f)  child.speed = 10.f;
        if (child.speed > 200.f) child.speed = 200.f;
        if (child.attack < 0.f)  child.attack = 0.f;
        if (child.attack > 50.f) child.attack = 50.f;
        if (child.senseRange < 20.f)  child.senseRange = 20.f;
        if (child.senseRange > 300.f) child.senseRange = 300.f;
        if (child.poisonResistance < 0.f) child.poisonResistance = 0.f;
        if (child.poisonResistance > 1.f) child.poisonResistance = 1.f;

        return child;
    }
};

//----------------------------------------------------------
// Entity(生物や植物の基底クラス)
//----------------------------------------------------------
class Entity {
public:
    virtual ~Entity() = default;
    virtual void update(float deltaTime) = 0;
    virtual void draw(sf::RenderWindow& window) = 0;
    virtual sf::Vector2f getPosition() const = 0;
    virtual bool isAlive() const = 0;
    virtual void onEaten() = 0;

    virtual float getCollisionRadius() const = 0;
};

//----------------------------------------------------------
// Plant(植物: 動かない)
//----------------------------------------------------------
class Plant : public Entity {
private:
    sf::CircleShape shape; 
    bool alive;
public:
    Plant(sf::Vector2f pos, float radius = 10.f, sf::Color color = sf::Color(120, 200, 120))
        : alive(true)
    {
        shape.setRadius(radius);
        shape.setOrigin(radius, radius);
        shape.setPosition(pos);
        shape.setFillColor(color);
    }

    void update(float /*deltaTime*/) override {
        // 動かない
    }

    void draw(sf::RenderWindow& window) override {
        if (alive) {
            window.draw(shape);
        }
    }

    sf::Vector2f getPosition() const override {
        return shape.getPosition();
    }

    bool isAlive() const override {
        return alive;
    }

    float getCollisionRadius() const override {
        return shape.getRadius();
    }

    void onEaten() override {
        alive = false;
    }
};

//----------------------------------------------------------
// Creature(動物的な生物) + GA(Genes) + Q学習
//----------------------------------------------------------
class Creature : public Entity {
private:
    //------------------------------------------------------
    // Q学習関連
    //------------------------------------------------------
    // 近くに「食料(弱い生物 or 植物)」がいるか, 
    // 近くに「強い捕食者」がいるか
    // → 2ビット(4状態)はそのまま使う例
    // 改良したい場合はここを拡張してもOK
    static const int NUM_STATES  = 4; // 00,01,10,11
    static const int NUM_ACTIONS = 4; // 前進,左旋回,右旋回,停止

    float Q[NUM_STATES][NUM_ACTIONS];

    float epsilon; // ε-greedy
    float alpha;   // 学習率
    float gamma;   // 割引率

    int   currentState;
    int   currentAction;

    //------------------------------------------------------
    // 遺伝子
    //------------------------------------------------------
    Genes genes;

    //------------------------------------------------------
    // 世代
    //------------------------------------------------------
    int generation;

    //------------------------------------------------------
    // 物理・描画関連
    //------------------------------------------------------
    sf::Vector2f position;
    float direction;   
    sf::CircleShape shape;
    bool alive;
    float energy;

    float reproductionCoolDown;

    //------------------------------------------------------
    // 他Entityへの参照リストを外部から受け取れるように
    // (observeStateで周囲のチェックをするため)
    //------------------------------------------------------
    const std::vector<std::shared_ptr<Entity>>* pAllEntities;

public:
    // コンストラクタで全Entityリスト参照も受け取る
    Creature(const Genes& g, sf::Vector2f pos, sf::Color color,
             int gen,
             const std::vector<std::shared_ptr<Entity>>* allEntities)
        : genes(g), generation(gen), position(pos),
          direction(getRandomFloat(0.f, 360.f)),
          alive(true), energy(50.f), reproductionCoolDown(0.f),
          pAllEntities(allEntities)
    {
        // Qテーブルを0初期化
        for(int s=0; s<NUM_STATES; s++){
            for(int a=0; a<NUM_ACTIONS; a++){
                Q[s][a] = 0.f;
            }
        }
        // ε-greedyのパラメータ
        epsilon = 0.2f;  // 必要に応じて下げてみる
        alpha   = 0.1f;
        gamma   = 0.9f;

        shape.setRadius(15.f);
        shape.setOrigin(15.f, 15.f);
        shape.setPosition(pos);
        color.a = 180;
        shape.setFillColor(color);

        direction = getRandomFloat(0.f, 360.f);
        currentState  = 0;
        currentAction = 0;
    }

    void update(float deltaTime) override {
        if(!alive) return;

        // 時間経過ペナルティ: 小さめに(生存コスト)
        float reward = -0.002f;

        // エネルギー消費
        energy -= deltaTime * 0.5f; 
        if (energy <= 0.f) {
            alive = false;
            // 死亡時ペナルティ
            reward -= 10.f;
            updateQ(reward);
            return;
        }

        // 前フレームの行動結果に対する Q値更新
        updateQ(reward);

        // 次状態観測 & 行動選択
        currentState = observeState();
        currentAction = selectAction(currentState);

        // 行動実行
        performAction(currentAction, deltaTime);

        // クールダウン時間計測
        if (reproductionCoolDown > 0.f) {
            reproductionCoolDown -= deltaTime;
        }
    }

    void draw(sf::RenderWindow& window) override {
        if (alive) {
            shape.setPosition(position);
            window.draw(shape);
        }
    }

    sf::Vector2f getPosition() const override {
        return position;
    }

    bool isAlive() const override {
        return alive;
    }

    float getCollisionRadius() const override {
        return shape.getRadius();
    }

    void onEaten() override {
        alive = false;
        // 捕食された時のペナルティ
        updateQ(-40.f);
    }

    // -----------------------------------------------------
    // GA + RL用
    // -----------------------------------------------------
    const Genes& getGenes() const {
        return genes;
    }

    float getAttackPower() const {
        return genes.attack;
    }

    bool isPoisonous() const {
        return genes.poison;
    }

    float getPoisonResistance() const {
        return genes.poisonResistance;
    }

    void addEnergy(float amount) {
        energy += amount;
    }

    bool canReproduce() const {
        return (energy > 50.f && reproductionCoolDown <= 0.f);
    }

    void resetReproductionCoolDown() {
        reproductionCoolDown = 5.f;
    }

    // 交配
    std::shared_ptr<Creature> reproduceWith(std::shared_ptr<Creature> other) {
        // 子に与えるエネルギー比: 0.6f
        float childEnergy = energy * 0.6f;
        energy *= 0.4f;

        Genes childGenes = Genes::crossoverAndMutate(this->genes, other->genes);

        sf::Color c1 = shape.getFillColor();
        sf::Color c2 = other->shape.getFillColor();
        sf::Uint8 red   = (sf::Uint8)std::min(255, (c1.r + c2.r)/2 + (rand()%11 - 5));
        sf::Uint8 green = (sf::Uint8)std::min(255, (c1.g + c2.g)/2 + (rand()%11 - 5));
        sf::Uint8 blue  = (sf::Uint8)std::min(255, (c1.b + c2.b)/2 + (rand()%11 - 5));
        sf::Color childColor(red, green, blue, 180);

        int newGen = std::max(this->generation, other->generation) + 1;

        auto child = std::make_shared<Creature>(childGenes, this->position, childColor, newGen, pAllEntities);
        child->energy = childEnergy;

        // Qテーブルの継承
        child->inheritQ(*this, *other);

        return child;
    }

    // ポジティブ報酬付与
    void givePositiveReward(float r) {
        updateQ(r);
    }

    // 世代を返す
    int getGeneration() const {
        return generation;
    }

    // Qテーブルの平均値を返す (学習状況をざっくり見る指標)
    float getAverageQ() const {
        float sum = 0.f;
        for(int s=0; s<NUM_STATES; s++){
            for(int a=0; a<NUM_ACTIONS; a++){
                sum += Q[s][a];
            }
        }
        return sum / (NUM_STATES * NUM_ACTIONS);
    }

    // “種族”名を返す
    std::string getSpeciesName() const {
        std::string speedCat;
        if(genes.speed < 60.f)         speedCat = "Slow";
        else if(genes.speed < 120.f)   speedCat = "Mid";
        else                           speedCat = "Fast";

        std::string attackCat;
        if(genes.attack < 10.f)        attackCat = "LowAtk";
        else if(genes.attack < 30.f)   attackCat = "MedAtk";
        else                           attackCat = "HighAtk";

        std::string poisonCat = genes.poison ? "Poison" : "NonPois";

        std::string legsStr = "Leg" + std::to_string(genes.legs);

        // 毒耐性のカテゴリ(3段階ぐらいにわけてみる)
        std::string resistCat;
        if(genes.poisonResistance < 0.33f)      resistCat = "LowRes";
        else if(genes.poisonResistance < 0.66f) resistCat = "MidRes";
        else                                    resistCat = "HighRes";

        // 例: "Slow_LowAtk_NonPois_Leg2_LowRes"
        return speedCat + "_" + attackCat + "_" + poisonCat + "_" + legsStr + "_" + resistCat;
    }

private:
    //------------------------------------------------------
    // 親の Q テーブルを引き継ぐ
    //------------------------------------------------------
    void inheritQ(const Creature& p1, const Creature& p2) {
        for(int s=0; s<NUM_STATES; s++){
            for(int a=0; a<NUM_ACTIONS; a++){
                float val = 0.5f * (p1.Q[s][a] + p2.Q[s][a]);
                val += getRandomFloat(-0.1f, 0.1f);

                if(val > 50.f) val = 50.f;
                if(val < -50.f) val = -50.f;

                this->Q[s][a] = val;
            }
        }
    }

    //------------------------------------------------------
    // 状態観測(周囲をチェック)
    //------------------------------------------------------
    int observeState() {
        bool foodNear = false;
        bool predatorNear = false;

        // senseRange 内に「食べられる(植物 or 攻撃力が自分より低いCreature)」がいるかチェック
        // senseRange 内に「自分より攻撃力が高いCreature」がいるかチェック
        if (!pAllEntities) {
            // もし参照がなければ従来通りランダムに
            if (rand()%100 < 8)  foodNear = true;
            if (rand()%100 < 5)  predatorNear = true;
        } else {
            float sr2 = genes.senseRange * genes.senseRange; // 2乗で比較
            for (auto& e : *pAllEntities) {
                if(!e->isAlive()) continue;
                if(e.get() == this) continue;

                // 距離判定
                float dist2 = distance2(this->position, e->getPosition());
                if(dist2 > sr2) continue; // 範囲外

                // Plant判定
                auto plant = std::dynamic_pointer_cast<Plant>(e);
                if(plant) {
                    foodNear = true;
                    continue;
                }
                // Creature判定
                auto c2 = std::dynamic_pointer_cast<Creature>(e);
                if(c2) {
                    if(c2->getAttackPower() < this->getAttackPower()) {
                        foodNear = true;
                    }
                    else if(c2->getAttackPower() > this->getAttackPower()) {
                        predatorNear = true;
                    }
                }
                // 一度 foodNear/predatorNear が true になれば十分
                if(foodNear && predatorNear) break;
            }
        }

        int s = 0;
        if(foodNear)     s |= 1; // bit0
        if(predatorNear) s |= 2; // bit1
        return s; // 0..3
    }

    //------------------------------------------------------
    // 行動選択(ε-greedy)
    //------------------------------------------------------
    int selectAction(int state) {
        if ((float)rand()/RAND_MAX < epsilon) {
            return rand() % NUM_ACTIONS;
        } else {
            float maxQ = Q[state][0];
            int bestA = 0;
            for(int a=1; a<NUM_ACTIONS; a++){
                if(Q[state][a] > maxQ){
                    maxQ = Q[state][a];
                    bestA = a;
                }
            }
            return bestA;
        }
    }

    //------------------------------------------------------
    // Q値更新
    //------------------------------------------------------
    void updateQ(float reward) {
        int s = currentState;
        int a = currentAction;

        int sNext = observeState();
        float maxQNext = Q[sNext][0];
        for(int i=1; i<NUM_ACTIONS; i++){
            if(Q[sNext][i] > maxQNext){
                maxQNext = Q[sNext][i];
            }
        }
        float oldQ = Q[s][a];
        float newQ = oldQ + alpha * (reward + gamma * maxQNext - oldQ);
        Q[s][a] = newQ;
    }

    //------------------------------------------------------
    // 行動実行
    //------------------------------------------------------
    void performAction(int action, float deltaTime) {
        float speedVal = genes.speed; 
        switch(action){
            case 0: {
                // 前進
                float rad = direction * 3.14159f / 180.f;
                position.x += cos(rad) * speedVal * deltaTime;
                position.y += sin(rad) * speedVal * deltaTime;
            } break;
            case 1: {
                // 左旋回
                direction -= 90.f * deltaTime;
            } break;
            case 2: {
                // 右旋回
                direction += 90.f * deltaTime;
            } break;
            case 3:
            default: {
                // 停止
            } break;
        }

        // 画面外に出ないようバウンド
        if (position.x < 0.f)    { position.x = 0.f;    direction += 180.f; }
        if (position.x > 800.f)  { position.x = 800.f;  direction += 180.f; }
        if (position.y < 0.f)    { position.y = 0.f;    direction += 180.f; }
        if (position.y > 600.f)  { position.y = 600.f;  direction += 180.f; }
    }
};

//----------------------------------------------------------
// 背景描画
//----------------------------------------------------------
void drawBackground(sf::RenderWindow& window, sf::Color color) {
    sf::RectangleShape rect;
    rect.setSize(sf::Vector2f(window.getSize().x, window.getSize().y));
    rect.setFillColor(color);
    rect.setPosition(0.f, 0.f);
    window.draw(rect);
}

//----------------------------------------------------------
// メイン
//----------------------------------------------------------
int main()
{
    srand((unsigned)time(NULL));

    sf::RenderWindow window(sf::VideoMode(800, 600), "GA + RL Evolution");
    window.setFramerateLimit(60);

    sf::Font font;
    if (!font.loadFromFile("./Roboto-VariableFont_wdth,wght.ttf")) {
        std::cerr << "Warning: Failed to load font. Text will not be visible.\n";
    }

    // Entityコンテナ
    std::vector<std::shared_ptr<Entity>> entities;

    // 初期Creature
    for(int i=0; i<8; i++){
        Genes g;
        g.speed           = getRandomFloat(30.f, 70.f);
        g.attack          = getRandomFloat(0.f, 5.f);
        g.poison          = (rand()%100 < 30);
        g.legs            = rand()%4 + 1;
        g.senseRange      = getRandomFloat(50.f, 150.f);
        g.poisonResistance= getRandomFloat(0.f, 1.f); // 初期ランダム

        float x = getRandomFloat(100.f, 700.f);
        float y = getRandomFloat(100.f, 500.f);

        sf::Color color(
            100 + rand()%156,
            100 + rand()%156,
            100 + rand()%156,
            180
        );
        auto c = std::make_shared<Creature>(g, sf::Vector2f(x,y), color, 0, &entities);
        entities.push_back(c);
    }

    // 初期Plant (30個)
    for(int i = 0; i < 30; i++){
        float x = getRandomFloat(50.f, 750.f);
        float y = getRandomFloat(50.f, 550.f);
        auto plant = std::make_shared<Plant>(sf::Vector2f(x,y));
        entities.push_back(plant);
    }

    // FPS計測
    sf::Clock clock;
    float fps = 0.f;
    float fpsTimer = 0.f;
    float fpsInterval = 0.5f;
    int frameCount = 0;

    while (window.isOpen()) {
        sf::Event ev;
        while (window.pollEvent(ev)) {
            if(ev.type == sf::Event::Closed) {
                window.close();
            }
        }

        float dt = clock.restart().asSeconds();
        fpsTimer += dt;
        frameCount++;
        if(fpsTimer >= fpsInterval){
            fps = frameCount / fpsTimer;
            frameCount = 0;
            fpsTimer = 0.f;
        }

        // Update
        for(auto& e : entities) {
            e->update(dt);
        }

        // 衝突・捕食判定
        for(auto& e1 : entities) {
            if(!e1->isAlive()) continue;
            auto c1 = std::dynamic_pointer_cast<Creature>(e1);
            if(!c1) continue;

            for(auto& e2 : entities) {
                if(!e2->isAlive()) continue;
                if(e1 == e2) continue;
                float dx = c1->getPosition().x - e2->getPosition().x;
                float dy = c1->getPosition().y - e2->getPosition().y;
                float dist = std::sqrt(dx*dx + dy*dy);
                if(dist < (c1->getCollisionRadius() + e2->getCollisionRadius())) {
                    // Plant
                    auto plant = std::dynamic_pointer_cast<Plant>(e2);
                    if(plant) {
                        plant->onEaten();
                        // 植物を食べた回復量
                        c1->addEnergy(15.f);
                        // 報酬
                        c1->givePositiveReward(5.f);
                    } else {
                        // Creature
                        auto c2 = std::dynamic_pointer_cast<Creature>(e2);
                        if(c2) {
                            // 攻撃力比較
                            float atk1 = c1->getAttackPower();
                            float atk2 = c2->getAttackPower();
                            if(atk1 > atk2) {
                                c2->onEaten();
                                c1->addEnergy(25.f);
                                c1->givePositiveReward(10.f);

                                // 毒ダメージ(軽減あり)
                                if(c2->isPoisonous()) {
                                    float poisonDmg = 12.f * (1.f - c1->getPoisonResistance());
                                    c1->addEnergy(-poisonDmg);
                                }
                            } else if(atk1 < atk2) {
                                c1->onEaten();
                                c2->addEnergy(25.f);
                                c2->givePositiveReward(10.f);

                                if(c1->isPoisonous()) {
                                    float poisonDmg = 12.f * (1.f - c2->getPoisonResistance());
                                    c2->addEnergy(-poisonDmg);
                                }
                            }
                            // 同じ攻撃力の場合は何もしない(共倒れなし)とする
                        }
                    }
                }
            }
        }

        // 増殖(交配)
        std::vector<std::shared_ptr<Entity>> newEntities;
        for(auto& e : entities) {
            if(!e->isAlive()) continue;
            auto c = std::dynamic_pointer_cast<Creature>(e);
            if(!c) continue;

            if(c->canReproduce()) {
                std::shared_ptr<Creature> partner = nullptr;
                for(auto& e2 : entities) {
                    if(!e2->isAlive()) continue;
                    if(e2 == e) continue;
                    auto c2 = std::dynamic_pointer_cast<Creature>(e2);
                    if(!c2) continue;
                    if(c2->canReproduce()) {
                        if(rand()%100 < 20) {
                            partner = c2;
                            break;
                        }
                    }
                }
                std::shared_ptr<Creature> child = nullptr;
                if(partner) {
                    child = c->reproduceWith(partner);
                    partner->resetReproductionCoolDown();
                } else {
                    // 単独増殖
                    child = c->reproduceWith(c);
                }
                c->resetReproductionCoolDown();
                newEntities.push_back(child);
            }
        }
        for(auto& ne : newEntities) {
            entities.push_back(ne);
        }

        // 死亡したEntityを削除
        entities.erase(
            std::remove_if(entities.begin(), entities.end(),
                [](std::shared_ptr<Entity> e){ return !e->isAlive(); }),
            entities.end()
        );

        // Plant不足なら補充
        int plantCount=0;
        for(auto& e : entities) {
            if(std::dynamic_pointer_cast<Plant>(e)) {
                plantCount++;
            }
        }
        if(plantCount < 15) {
            for(int i=0; i<5; i++){
                float x = getRandomFloat(50.f, 750.f);
                float y = getRandomFloat(50.f, 550.f);
                auto plant = std::make_shared<Plant>(sf::Vector2f(x,y));
                entities.push_back(plant);
            }
        }

        // 描画
        window.clear();
        drawBackground(window, sf::Color(220,220,220));

        for(auto& e : entities){
            e->draw(window);
        }

        // UI (FPS, 数表示, 世代, 学習状況, 種族別個体数)
        {
            // UIパネル
            sf::RectangleShape uiPanel(sf::Vector2f(220.f, 320.f));
            uiPanel.setFillColor(sf::Color(255,255,255,180));
            uiPanel.setPosition(20.f,20.f);
            window.draw(uiPanel);

            int creatureCount=0;
            float totalQ = 0.f;
            int   qCount = 0;
            int maxGen = 0;
            std::map<std::string, int> speciesCount;

            for(auto& e : entities){
                auto c = std::dynamic_pointer_cast<Creature>(e);
                if(c) {
                    creatureCount++;
                    float qval = c->getAverageQ();
                    totalQ += qval;
                    qCount++;
                    if(c->getGeneration() > maxGen) {
                        maxGen = c->getGeneration();
                    }
                    speciesCount[c->getSpeciesName()]++;
                }
            }

            float avgQ = (qCount > 0) ? (totalQ / qCount) : 0.f;

            std::string info;
            info += "FPS: " + std::to_string((int)fps) + "\n";
            info += "Creature: " + std::to_string(creatureCount) + "\n";
            info += "Plant:    " + std::to_string(plantCount) + "\n";
            info += "Max Gen:  " + std::to_string(maxGen) + "\n";
            info += "Avg Q:    " + std::to_string(avgQ) + "\n";

            info += "\n--- Species Count ---\n";
            for(const auto& kv : speciesCount) {
                info += kv.first + ": " + std::to_string(kv.second) + "\n";
            }

            if (font.getInfo().family != "") {
                sf::Text txt;
                txt.setFont(font);
                txt.setString(info);
                txt.setCharacterSize(14);
                txt.setFillColor(sf::Color::Black);
                txt.setPosition(30.f, 28.f);
                window.draw(txt);
            }
        }

        window.display();
    }

    return 0;
}
