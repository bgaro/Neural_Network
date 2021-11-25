
float unit_step(float x)
{
    return x >= 0 ? 1 : 0;
}

float reLU(float x)
{
    return x >= 0 ? x : 0;
}
