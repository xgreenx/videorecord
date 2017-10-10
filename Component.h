//
// Created by green on 23.09.17.
//

#ifndef VIDEORECORD_COMPONENT_H
#define VIDEORECORD_COMPONENT_H

#include <utility>

typedef std::pair<int, int> Poin;

class Component {
public:
    Poin Left;
    Poin Right;
    Poin Top;
    Poin Down;

    int count = 0;

    Component() : Component(-1, -1)
    {

    }

    Component(int i, int j)
    {
        Left = {i, j};
        Right = {i, j};
        Top = {i, j};
        Down = {i, j};
        count = 1;
    }
};


#endif //VIDEORECORD_COMPONENT_H
